import builtins
import importlib
import json
import os
import re
import sys
import zipfile
from collections import Counter
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
    clone_model_ir_with_float32,
    infer_model_ir_logical_layouts,
    normalize_logical_layout,
    optimize_redundant_transpose_operators,
    prune_identity_cast_operators,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.pytorch_package_runtime import (
    _align_tensor_to_target_shape,
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
    _build_metadata_payload,
    _build_tensor_var_name_map,
    _build_torchscript_example_inputs,
    _export_runtime_wrapper_package_from_model_ir,
    _make_tensor_storage_name_map,
    _merge_reference_public_boundary_metadata,
    _propagate_pytorch_friendly_layouts,
    _reject_residual_layout_transposes,
    _remove_redundant_layout_transposes,
    _should_prefer_tflite_backed_package,
    _write_native_model_file,
    export_dynamo_onnx_from_generated_package,
    export_exported_program_from_generated_package,
    export_torchscript_from_generated_package,
    export_pytorch_package_from_model_ir,
    normalize_model_ir_for_pytorch_channel_first,
    prepare_model_ir_for_native_pytorch,
    validate_channel_first_exportability,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module
from onnx2tf.tflite_builder.tflite_importer import import_model_ir_from_tflite


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


def _make_if_axis0_tensor_mux_model_ir() -> ModelIR:
    model_ir = ModelIR(name="if_axis0_tensor_mux_export")
    model_ir.inputs = ["If_p1_input1", "If_p1_input2"]
    model_ir.outputs = ["If_p1_output"]

    model_ir.tensors["If_p1_input1"] = TensorIR(
        name="If_p1_input1",
        dtype="FLOAT32",
        shape=[1, 100],
        shape_signature=[1, 100],
    )
    model_ir.tensors["If_p1_input2"] = TensorIR(
        name="If_p1_input2",
        dtype="FLOAT32",
        shape=[2, 100],
        shape_signature=[2, 100],
    )
    model_ir.tensors["wa_reduce_axes"] = TensorIR(
        name="wa_reduce_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([0, 1], dtype=np.int32),
    )
    model_ir.tensors["wa_reduce_1_axes"] = TensorIR(
        name="wa_reduce_1_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([0, 1], dtype=np.int32),
    )
    model_ir.tensors["wa_ReduceSum_output_0"] = TensorIR(
        name="wa_ReduceSum_output_0",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["wa_ReduceSum_1_output_0"] = TensorIR(
        name="wa_ReduceSum_1_output_0",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["wa_Greater_output_0"] = TensorIR(
        name="wa_Greater_output_0",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["then_bias"] = TensorIR(
        name="then_bias",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0], dtype=np.float32),
    )
    model_ir.tensors["else_bias"] = TensorIR(
        name="else_bias",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2.0], dtype=np.float32),
    )
    model_ir.tensors["If_p1_output_if_then"] = TensorIR(
        name="If_p1_output_if_then",
        dtype="FLOAT32",
        shape=[1, 100],
        shape_signature=[1, 100],
    )
    model_ir.tensors["If_p1_output_if_else"] = TensorIR(
        name="If_p1_output_if_else",
        dtype="FLOAT32",
        shape=[2, 100],
        shape_signature=[2, 100],
    )
    model_ir.tensors["If_p1_output_if_merged"] = TensorIR(
        name="If_p1_output_if_merged",
        dtype="FLOAT32",
        shape=[3, 100],
        shape_signature=[3, 100],
    )
    model_ir.tensors["If_p1_output_if_cond_i32"] = TensorIR(
        name="If_p1_output_if_cond_i32",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["If_p1_output_if_one"] = TensorIR(
        name="If_p1_output_if_one",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray(1, dtype=np.int32),
    )
    model_ir.tensors["If_p1_output_if_not_cond_i32"] = TensorIR(
        name="If_p1_output_if_not_cond_i32",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["If_p1_output_if_then_first_dim"] = TensorIR(
        name="If_p1_output_if_then_first_dim",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray(1, dtype=np.int32),
    )
    model_ir.tensors["If_p1_output_if_else_first_dim"] = TensorIR(
        name="If_p1_output_if_else_first_dim",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray(2, dtype=np.int32),
    )
    model_ir.tensors["If_p1_output_if_begin_axis0"] = TensorIR(
        name="If_p1_output_if_begin_axis0",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["If_p1_output_if_size_then"] = TensorIR(
        name="If_p1_output_if_size_then",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["If_p1_output_if_size_else"] = TensorIR(
        name="If_p1_output_if_size_else",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["If_p1_output_if_size_axis0"] = TensorIR(
        name="If_p1_output_if_size_axis0",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["If_p1_output_if_begin_tail"] = TensorIR(
        name="If_p1_output_if_begin_tail",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["If_p1_output_if_size_tail"] = TensorIR(
        name="If_p1_output_if_size_tail",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([100], dtype=np.int32),
    )
    model_ir.tensors["If_p1_output_if_begin"] = TensorIR(
        name="If_p1_output_if_begin",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
    )
    model_ir.tensors["If_p1_output_if_size"] = TensorIR(
        name="If_p1_output_if_size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
    )
    model_ir.tensors["If_p1_output"] = TensorIR(
        name="If_p1_output",
        dtype="FLOAT32",
        shape=[1, 100],
        shape_signature=[-1, 100],
    )

    model_ir.operators.extend(
        [
            OperatorIR(op_type="SUM", inputs=["If_p1_input1", "wa_reduce_axes"], outputs=["wa_ReduceSum_output_0"], options={"keepDims": False}),
            OperatorIR(op_type="SUM", inputs=["If_p1_input2", "wa_reduce_1_axes"], outputs=["wa_ReduceSum_1_output_0"], options={"keepDims": False}),
            OperatorIR(op_type="GREATER", inputs=["wa_ReduceSum_output_0", "wa_ReduceSum_1_output_0"], outputs=["wa_Greater_output_0"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="ADD", inputs=["If_p1_input1", "then_bias"], outputs=["If_p1_output_if_then"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="ADD", inputs=["If_p1_input2", "else_bias"], outputs=["If_p1_output_if_else"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="CONCATENATION", inputs=["If_p1_output_if_then", "If_p1_output_if_else"], outputs=["If_p1_output_if_merged"], options={"axis": 0, "fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="CAST", inputs=["wa_Greater_output_0"], outputs=["If_p1_output_if_cond_i32"], options={"inDataType": "BOOL", "outDataType": "INT32"}),
            OperatorIR(op_type="SUB", inputs=["If_p1_output_if_one", "If_p1_output_if_cond_i32"], outputs=["If_p1_output_if_not_cond_i32"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="MUL", inputs=["If_p1_output_if_not_cond_i32", "If_p1_output_if_then_first_dim"], outputs=["If_p1_output_if_begin_axis0"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="MUL", inputs=["If_p1_output_if_cond_i32", "If_p1_output_if_then_first_dim"], outputs=["If_p1_output_if_size_then"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="MUL", inputs=["If_p1_output_if_not_cond_i32", "If_p1_output_if_else_first_dim"], outputs=["If_p1_output_if_size_else"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="ADD", inputs=["If_p1_output_if_size_then", "If_p1_output_if_size_else"], outputs=["If_p1_output_if_size_axis0"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="CONCATENATION", inputs=["If_p1_output_if_begin_axis0", "If_p1_output_if_begin_tail"], outputs=["If_p1_output_if_begin"], options={"axis": 0, "fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="CONCATENATION", inputs=["If_p1_output_if_size_axis0", "If_p1_output_if_size_tail"], outputs=["If_p1_output_if_size"], options={"axis": 0, "fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="SLICE", inputs=["If_p1_output_if_merged", "If_p1_output_if_begin", "If_p1_output_if_size"], outputs=["If_p1_output"], options={}),
        ]
    )
    return model_ir


def _make_conv3d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 3, 4])
    w = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [2, 1, 2, 2, 2],
        (np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float32) / 5.0).tolist(),
    )
    b = helper.make_tensor(
        "b",
        TensorProto.FLOAT,
        [2],
        np.asarray([0.25, -0.5], dtype=np.float32).tolist(),
    )
    node = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        name="Conv3DNode",
        kernel_shape=[2, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
        strides=[1, 1, 1],
    )
    graph = helper.make_graph([node], "conv3d_graph", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    return model


def _make_conv_transpose3d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 3, 3])
    w = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [1, 2, 2, 2, 2],
        (np.arange(1 * 2 * 2 * 2 * 2, dtype=np.float32) / 7.0).tolist(),
    )
    b = helper.make_tensor(
        "b",
        TensorProto.FLOAT,
        [2],
        np.asarray([0.5, -0.25], dtype=np.float32).tolist(),
    )
    node = helper.make_node(
        "ConvTranspose",
        ["x", "w", "b"],
        ["y"],
        name="ConvTranspose3DNode",
        kernel_shape=[2, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
        strides=[1, 1, 1],
    )
    graph = helper.make_graph([node], "conv_transpose3d_graph", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    return model


def _make_conv_transpose2d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 3])
    w = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [1, 2, 2, 2],
        (np.arange(1 * 2 * 2 * 2, dtype=np.float32) / 7.0).tolist(),
    )
    b = helper.make_tensor(
        "b",
        TensorProto.FLOAT,
        [2],
        np.asarray([0.5, -0.25], dtype=np.float32).tolist(),
    )
    node = helper.make_node(
        "ConvTranspose",
        ["x", "w", "b"],
        ["y"],
        name="ConvTranspose2DNode",
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "conv_transpose2d_graph", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    return model


def _make_nhwc_mlp_bridge_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 4])
    conv_w = helper.make_tensor(
        "conv_w",
        TensorProto.FLOAT,
        [4, 3, 1, 1],
        (np.arange(4 * 3, dtype=np.float32) / 10.0).tolist(),
    )
    conv_b = helper.make_tensor(
        "conv_b",
        TensorProto.FLOAT,
        [4],
        np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32).tolist(),
    )
    ln_scale = helper.make_tensor(
        "ln_scale",
        TensorProto.FLOAT,
        [4],
        np.asarray([1.0, 0.9, 1.1, 1.2], dtype=np.float32).tolist(),
    )
    ln_bias = helper.make_tensor(
        "ln_bias",
        TensorProto.FLOAT,
        [4],
        np.asarray([0.05, -0.1, 0.15, -0.2], dtype=np.float32).tolist(),
    )
    fc1_w = helper.make_tensor(
        "fc1_w",
        TensorProto.FLOAT,
        [4, 6],
        (np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 15.0).reshape(-1).tolist(),
    )
    fc1_b = helper.make_tensor(
        "fc1_b",
        TensorProto.FLOAT,
        [6],
        np.asarray([0.2, -0.1, 0.05, -0.05, 0.15, -0.2], dtype=np.float32).tolist(),
    )
    fc2_w = helper.make_tensor(
        "fc2_w",
        TensorProto.FLOAT,
        [6, 4],
        (np.arange(6 * 4, dtype=np.float32).reshape(6, 4) / 20.0).reshape(-1).tolist(),
    )
    fc2_b = helper.make_tensor(
        "fc2_b",
        TensorProto.FLOAT,
        [4],
        np.asarray([-0.05, 0.1, -0.15, 0.2], dtype=np.float32).tolist(),
    )
    nodes = [
        helper.make_node("Conv", ["x", "conv_w", "conv_b"], ["conv"], name="StemConv"),
        helper.make_node("Transpose", ["conv"], ["conv_nhwc"], name="ToNHWC", perm=[0, 2, 3, 1]),
        helper.make_node(
            "LayerNormalization",
            ["conv_nhwc", "ln_scale", "ln_bias"],
            ["ln"],
            name="LN",
            axis=-1,
            epsilon=1e-5,
        ),
        helper.make_node("MatMul", ["ln", "fc1_w"], ["fc1"], name="FC1"),
        helper.make_node("Add", ["fc1", "fc1_b"], ["fc1_bias"], name="FC1Bias"),
        helper.make_node("Gelu", ["fc1_bias"], ["gelu"], name="GELU"),
        helper.make_node("MatMul", ["gelu", "fc2_w"], ["fc2"], name="FC2"),
        helper.make_node("Add", ["fc2", "fc2_b"], ["y"], name="FC2Bias"),
    ]
    graph = helper.make_graph(
        nodes,
        "nhwc_mlp_bridge_graph",
        [x],
        [y],
        initializer=[conv_w, conv_b, ln_scale, ln_bias, fc1_w, fc1_b, fc2_w, fc2_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model


def _make_nhwc_depthwise_bridge_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 6])
    stem_w = helper.make_tensor(
        "stem_w",
        TensorProto.FLOAT,
        [4, 3, 1, 1],
        (np.arange(4 * 3, dtype=np.float32) / 9.0).tolist(),
    )
    stem_b = helper.make_tensor(
        "stem_b",
        TensorProto.FLOAT,
        [4],
        np.asarray([0.0, 0.1, -0.1, 0.2], dtype=np.float32).tolist(),
    )
    ln_scale = helper.make_tensor(
        "ln_scale_dw",
        TensorProto.FLOAT,
        [4],
        np.asarray([1.0, 1.1, 0.9, 1.2], dtype=np.float32).tolist(),
    )
    ln_bias = helper.make_tensor(
        "ln_bias_dw",
        TensorProto.FLOAT,
        [4],
        np.asarray([0.05, -0.05, 0.1, -0.1], dtype=np.float32).tolist(),
    )
    dw_w = helper.make_tensor(
        "dw_w",
        TensorProto.FLOAT,
        [4, 1, 3, 3],
        (np.arange(4 * 3 * 3, dtype=np.float32) / 30.0).tolist(),
    )
    dw_b = helper.make_tensor(
        "dw_b",
        TensorProto.FLOAT,
        [4],
        np.asarray([0.0, 0.05, -0.05, 0.1], dtype=np.float32).tolist(),
    )
    fc_w = helper.make_tensor(
        "dw_fc_w",
        TensorProto.FLOAT,
        [4, 6],
        (np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 12.0).reshape(-1).tolist(),
    )
    fc_b = helper.make_tensor(
        "dw_fc_b",
        TensorProto.FLOAT,
        [6],
        np.asarray([0.1, -0.2, 0.3, -0.1, 0.2, -0.3], dtype=np.float32).tolist(),
    )
    nodes = [
        helper.make_node("Conv", ["x", "stem_w", "stem_b"], ["stem"], name="StemConv"),
        helper.make_node("Transpose", ["stem"], ["stem_nhwc"], name="StemToNHWC", perm=[0, 2, 3, 1]),
        helper.make_node(
            "LayerNormalization",
            ["stem_nhwc", "ln_scale_dw", "ln_bias_dw"],
            ["ln_dw"],
            name="LNDW",
            axis=-1,
            epsilon=1e-5,
        ),
        helper.make_node("Transpose", ["ln_dw"], ["ln_dw_nchw"], name="ToNCHW", perm=[0, 3, 1, 2]),
        helper.make_node(
            "Conv",
            ["ln_dw_nchw", "dw_w", "dw_b"],
            ["dw"],
            name="Depthwise",
            group=4,
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Transpose", ["dw"], ["dw_nhwc"], name="DWToNHWC", perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", ["dw_nhwc", "dw_fc_w"], ["fc_dw"], name="FC"),
        helper.make_node("Add", ["fc_dw", "dw_fc_b"], ["y"], name="FCBias"),
    ]
    graph = helper.make_graph(
        nodes,
        "nhwc_depthwise_bridge_graph",
        [x],
        [y],
        initializer=[stem_w, stem_b, ln_scale, ln_bias, dw_w, dw_b, fc_w, fc_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model


def _make_small_face_liveness_style_model() -> onnx.ModelProto:
    rng = np.random.default_rng(23)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    initializers = [
        helper.make_tensor(
            "w0",
            TensorProto.FLOAT,
            [2, 3, 3, 3],
            rng.standard_normal((2, 3, 3, 3)).astype(np.float32).reshape(-1).tolist(),
        ),
        helper.make_tensor(
            "b0",
            TensorProto.FLOAT,
            [2],
            rng.standard_normal((2,)).astype(np.float32).tolist(),
        ),
        helper.make_tensor(
            "w1",
            TensorProto.FLOAT,
            [2, 2, 3, 3],
            rng.standard_normal((2, 2, 3, 3)).astype(np.float32).reshape(-1).tolist(),
        ),
        helper.make_tensor(
            "b1",
            TensorProto.FLOAT,
            [2],
            rng.standard_normal((2,)).astype(np.float32).tolist(),
        ),
        helper.make_tensor(
            "w2",
            TensorProto.FLOAT,
            [4, 32],
            rng.standard_normal((4, 32)).astype(np.float32).reshape(-1).tolist(),
        ),
        helper.make_tensor(
            "b2",
            TensorProto.FLOAT,
            [4],
            rng.standard_normal((4,)).astype(np.float32).tolist(),
        ),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "w0", "b0"], ["conv0"], name="Conv0", pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Relu", ["conv0"], ["relu0"], name="Relu0"),
        helper.make_node("Conv", ["relu0", "w1", "b1"], ["conv1"], name="Conv1", pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Relu", ["conv1"], ["relu1"], name="Relu1"),
        helper.make_node("MaxPool", ["relu1"], ["pool"], name="Pool0", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["pool"], ["flat"], name="Flatten0", axis=1),
        helper.make_node("Gemm", ["flat", "w2", "b2"], ["logits"], name="Gemm0", transB=1),
        helper.make_node("Softmax", ["logits"], ["output"], name="Softmax0", axis=1),
    ]
    graph = helper.make_graph(nodes, "small_face_liveness_style", [x], [y], initializer=initializers)
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


def _make_cumsum_model_ir(*, axis: int = 1, exclusive: bool = False, reverse: bool = False) -> ModelIR:
    model_ir = ModelIR(name="cumsum_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[2, 3], shape_signature=[2, 3])
    model_ir.tensors["axis"] = TensorIR(
        name="axis",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(axis, dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2, 3], shape_signature=[2, 3])
    model_ir.operators.append(
        OperatorIR(
            op_type="CUMSUM",
            inputs=["x", "axis"],
            outputs=["y"],
            options={"exclusive": exclusive, "reverse": reverse},
        )
    )
    return model_ir


def _make_bidirectional_sequence_lstm_model_ir() -> ModelIR:
    rng = np.random.default_rng(211)
    time_steps = 3
    batch = 1
    input_size = 2
    hidden_size = 2

    model_ir = ModelIR(name="bidirectional_sequence_lstm_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[time_steps, batch, input_size],
        shape_signature=[time_steps, batch, input_size],
        logical_layout="NWC",
    )

    for prefix in ["fw", "bw"]:
        for suffix in ["wi", "wf", "wc", "wo"]:
            name = f"{prefix}_{suffix}"
            model_ir.tensors[name] = TensorIR(
                name=name,
                dtype="FLOAT32",
                shape=[hidden_size, input_size],
                shape_signature=[hidden_size, input_size],
                data=rng.standard_normal((hidden_size, input_size)).astype(np.float32),
            )
        for suffix in ["ri", "rf", "rc", "ro"]:
            name = f"{prefix}_{suffix}"
            model_ir.tensors[name] = TensorIR(
                name=name,
                dtype="FLOAT32",
                shape=[hidden_size, hidden_size],
                shape_signature=[hidden_size, hidden_size],
                data=rng.standard_normal((hidden_size, hidden_size)).astype(np.float32),
            )
        for suffix in ["bi", "bf", "bc", "bo"]:
            name = f"{prefix}_{suffix}"
            model_ir.tensors[name] = TensorIR(
                name=name,
                dtype="FLOAT32",
                shape=[hidden_size],
                shape_signature=[hidden_size],
                data=rng.standard_normal((hidden_size,)).astype(np.float32),
            )

    model_ir.tensors["fw_h0"] = TensorIR(
        name="fw_h0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
        data=np.zeros((batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["fw_c0"] = TensorIR(
        name="fw_c0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
        data=np.zeros((batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["bw_h0"] = TensorIR(
        name="bw_h0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
        data=np.zeros((batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["bw_c0"] = TensorIR(
        name="bw_c0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
        data=np.zeros((batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[time_steps, batch, hidden_size * 2],
        shape_signature=[time_steps, batch, hidden_size * 2],
        logical_layout="NWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="BIDIRECTIONAL_SEQUENCE_LSTM",
            inputs=[
                "x",
                "fw_wi", "fw_wf", "fw_wc", "fw_wo",
                "fw_ri", "fw_rf", "fw_rc", "fw_ro",
                "", "", "",
                "fw_bi", "fw_bf", "fw_bc", "fw_bo",
                "", "",
                "bw_wi", "bw_wf", "bw_wc", "bw_wo",
                "bw_ri", "bw_rf", "bw_rc", "bw_ro",
                "", "", "",
                "bw_bi", "bw_bf", "bw_bc", "bw_bo",
                "", "",
                "fw_h0", "fw_c0", "bw_h0", "bw_c0",
                "", "", "", "", "", "", "", "", "",
            ],
            outputs=["y"],
            options={
                "fusedActivationFunction": "TANH",
                "cellClip": 0.0,
                "projClip": 0.0,
                "mergeOutputs": True,
                "timeMajor": True,
                "asymmetricQuantizeInputs": False,
            },
        )
    )
    return model_ir


def _make_compact_bidirectional_sequence_lstm_model_ir() -> ModelIR:
    model_ir = _make_bidirectional_sequence_lstm_model_ir()
    model_ir.operators = [
        OperatorIR(
            op_type="BIDIRECTIONAL_SEQUENCE_LSTM",
            inputs=[
                "x",
                "fw_wi", "fw_wf", "fw_wc", "fw_wo",
                "fw_ri", "fw_rf", "fw_rc", "fw_ro",
                "fw_bi", "fw_bf", "fw_bc", "fw_bo",
                "bw_wi", "bw_wf", "bw_wc", "bw_wo",
                "bw_ri", "bw_rf", "bw_rc", "bw_ro",
                "bw_bi", "bw_bf", "bw_bc", "bw_bo",
                "fw_h0", "fw_c0", "bw_h0", "bw_c0",
            ],
            outputs=["y"],
            options={
                "fusedActivationFunction": "TANH",
                "cellClip": 0.0,
                "projClip": 0.0,
                "mergeOutputs": True,
                "timeMajor": True,
                "asymmetricQuantizeInputs": False,
            },
        )
    ]
    return model_ir


def _make_reverse_lstm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 1, 1, 2])
    w = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [1, 8, 2],
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
        ).reshape(-1).tolist(),
    )
    r = helper.make_tensor(
        "R",
        TensorProto.FLOAT,
        [1, 8, 2],
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
        ).reshape(-1).tolist(),
    )
    b = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [1, 16],
        np.asarray(
            [[0.01, -0.02, 0.03, 0.01, -0.01, 0.00, 0.02, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ).reshape(-1).tolist(),
    )
    initial_h = helper.make_tensor(
        "initial_h",
        TensorProto.FLOAT,
        [1, 1, 2],
        np.zeros((1, 1, 2), dtype=np.float32).reshape(-1).tolist(),
    )
    initial_c = helper.make_tensor(
        "initial_c",
        TensorProto.FLOAT,
        [1, 1, 2],
        np.zeros((1, 1, 2), dtype=np.float32).reshape(-1).tolist(),
    )
    lstm = helper.make_node(
        "LSTM",
        ["x", "W", "R", "B", "", "initial_h", "initial_c"],
        ["y"],
        name="ReverseLSTMNode",
        hidden_size=2,
        direction="reverse",
    )
    graph = helper.make_graph(
        [lstm],
        "reverse_lstm_graph",
        [x],
        [y],
        initializer=[w, r, b, initial_h, initial_c],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    return model


def _make_bidirectional_sequence_rnn_model_ir() -> ModelIR:
    rng = np.random.default_rng(17)
    time_steps = 3
    batch = 1
    input_size = 2
    hidden_size = 3
    model_ir = ModelIR(name="bidirectional_sequence_rnn_native")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[time_steps, batch, input_size],
        shape_signature=[time_steps, batch, input_size],
        logical_layout="NWC",
    )
    model_ir.tensors["h0_3d"] = TensorIR(
        name="h0_3d",
        dtype="FLOAT32",
        shape=[1, batch, hidden_size],
        shape_signature=[1, batch, hidden_size],
        data=np.zeros((1, batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["h0_shape"] = TensorIR(
        name="h0_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([batch, hidden_size], dtype=np.int32),
    )
    model_ir.tensors["h0"] = TensorIR(
        name="h0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
    )
    model_ir.tensors["reverse_axis"] = TensorIR(
        name="reverse_axis",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["x_rev"] = TensorIR(
        name="x_rev",
        dtype="FLOAT32",
        shape=[time_steps, batch, input_size],
        shape_signature=[time_steps, batch, input_size],
        logical_layout="NWC",
    )
    for prefix in ["fw", "bw"]:
        model_ir.tensors[f"{prefix}_w"] = TensorIR(
            name=f"{prefix}_w",
            dtype="FLOAT32",
            shape=[hidden_size, input_size],
            shape_signature=[hidden_size, input_size],
            data=rng.standard_normal((hidden_size, input_size)).astype(np.float32),
        )
        model_ir.tensors[f"{prefix}_r"] = TensorIR(
            name=f"{prefix}_r",
            dtype="FLOAT32",
            shape=[hidden_size, hidden_size],
            shape_signature=[hidden_size, hidden_size],
            data=rng.standard_normal((hidden_size, hidden_size)).astype(np.float32),
        )
        model_ir.tensors[f"{prefix}_b"] = TensorIR(
            name=f"{prefix}_b",
            dtype="FLOAT32",
            shape=[hidden_size],
            shape_signature=[hidden_size],
            data=rng.standard_normal((hidden_size,)).astype(np.float32),
        )
        model_ir.tensors[f"{prefix}_y_raw"] = TensorIR(
            name=f"{prefix}_y_raw",
            dtype="FLOAT32",
            shape=[time_steps, batch, hidden_size],
            shape_signature=[time_steps, batch, hidden_size],
            logical_layout="NWC",
        )
    model_ir.tensors["bw_y"] = TensorIR(
        name="bw_y",
        dtype="FLOAT32",
        shape=[time_steps, batch, hidden_size],
        shape_signature=[time_steps, batch, hidden_size],
        logical_layout="NWC",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[time_steps, batch, hidden_size * 2],
        shape_signature=[time_steps, batch, hidden_size * 2],
        logical_layout="NWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="RESHAPE",
                inputs=["h0_3d", "h0_shape"],
                outputs=["h0"],
                options={"newShape": [batch, hidden_size]},
            ),
            OperatorIR(
                op_type="REVERSE_V2",
                inputs=["x", "reverse_axis"],
                outputs=["x_rev"],
                options={},
            ),
            OperatorIR(
                op_type="UNIDIRECTIONAL_SEQUENCE_RNN",
                inputs=["x", "fw_w", "fw_r", "fw_b", "h0"],
                outputs=["fw_y_raw"],
                options={
                    "timeMajor": True,
                    "fusedActivationFunction": "RELU",
                    "asymmetricQuantizeInputs": False,
                },
            ),
            OperatorIR(
                op_type="UNIDIRECTIONAL_SEQUENCE_RNN",
                inputs=["x_rev", "bw_w", "bw_r", "bw_b", "h0"],
                outputs=["bw_y_raw"],
                options={
                    "timeMajor": True,
                    "fusedActivationFunction": "RELU",
                    "asymmetricQuantizeInputs": False,
                },
            ),
            OperatorIR(
                op_type="REVERSE_V2",
                inputs=["bw_y_raw", "reverse_axis"],
                outputs=["bw_y"],
                options={},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["fw_y_raw", "bw_y"],
                outputs=["y"],
                options={"axis": 2, "fusedActivationFunction": "NONE"},
            ),
        ]
    )
    return model_ir


def _make_feature_last_residual_norm_model_ir() -> ModelIR:
    model_ir = ModelIR(name="feature_last_residual_norm")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[2, 3, 4],
        shape_signature=[2, 3, 4],
        logical_layout="NWC",
    )
    model_ir.tensors["perm_pre"] = TensorIR(
        name="perm_pre",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 0, 2], dtype=np.int32),
    )
    model_ir.tensors["t0"] = TensorIR(
        name="t0",
        dtype="FLOAT32",
        shape=[3, 2, 4],
        shape_signature=[3, 2, 4],
        logical_layout="NWC",
    )
    model_ir.tensors["reshape_flat_shape"] = TensorIR(
        name="reshape_flat_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([3, 8], dtype=np.int32),
    )
    model_ir.tensors["flat"] = TensorIR(
        name="flat",
        dtype="FLOAT32",
        shape=[3, 8],
        shape_signature=[3, 8],
    )
    model_ir.tensors["fc_w"] = TensorIR(
        name="fc_w",
        dtype="FLOAT32",
        shape=[8, 8],
        shape_signature=[8, 8],
        data=np.eye(8, dtype=np.float32),
    )
    model_ir.tensors["fc_b"] = TensorIR(
        name="fc_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.linspace(-0.4, 0.3, 8, dtype=np.float32),
    )
    model_ir.tensors["fc"] = TensorIR(
        name="fc",
        dtype="FLOAT32",
        shape=[3, 8],
        shape_signature=[3, 8],
    )
    model_ir.tensors["reshape_seq_shape"] = TensorIR(
        name="reshape_seq_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([3, 1, 8], dtype=np.int32),
    )
    model_ir.tensors["seq"] = TensorIR(
        name="seq",
        dtype="FLOAT32",
        shape=[3, 1, 8],
        shape_signature=[3, 1, 8],
        logical_layout="NWC",
    )
    model_ir.tensors["perm_post"] = TensorIR(
        name="perm_post",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 0, 2], dtype=np.int32),
    )
    model_ir.tensors["residual"] = TensorIR(
        name="residual",
        dtype="FLOAT32",
        shape=[1, 3, 8],
        shape_signature=[1, 3, 8],
        logical_layout="NWC",
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1, 3, 8],
        shape_signature=[1, 3, 8],
        data=(np.arange(24, dtype=np.float32).reshape(1, 3, 8) / 20.0),
        logical_layout="NWC",
    )
    model_ir.tensors["sum"] = TensorIR(
        name="sum",
        dtype="FLOAT32",
        shape=[1, 3, 8],
        shape_signature=[1, 3, 8],
        logical_layout="NWC",
    )
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 1],
        shape_signature=[1, 3, 1],
        logical_layout="NWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_pre"], outputs=["t0"], options={}),
            OperatorIR(op_type="RESHAPE", inputs=["t0", "reshape_flat_shape"], outputs=["flat"], options={"newShape": [3, 8]}),
            OperatorIR(op_type="FULLY_CONNECTED", inputs=["flat", "fc_w", "fc_b"], outputs=["fc"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="RESHAPE", inputs=["fc", "reshape_seq_shape"], outputs=["seq"], options={"newShape": [3, 1, 8]}),
            OperatorIR(op_type="TRANSPOSE", inputs=["seq", "perm_post"], outputs=["residual"], options={}),
            OperatorIR(op_type="ADD", inputs=["bias", "residual"], outputs=["sum"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="MEAN", inputs=["sum", "axes"], outputs=["y"], options={"keepDims": True}),
        ]
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


def _make_reduce_mean_constant_axes_model_ir() -> ModelIR:
    model_ir = ModelIR(name="reduce_mean_constant_axes_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 5],
        shape_signature=[1, 3, 4, 5],
        logical_layout="NCHW",
    )
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 1, 1],
        shape_signature=[1, 3, 1, 1],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="MEAN", inputs=["x", "axes"], outputs=["y"], options={"keepDims": True})
    )
    return model_ir


def _make_non_max_suppression_v4_model_ir() -> ModelIR:
    model_ir = ModelIR(name="non_max_suppression_v4_model_ir")
    model_ir.inputs = ["boxes", "scores"]
    model_ir.outputs = ["selected_indices", "valid_count"]
    model_ir.tensors["boxes"] = TensorIR(
        name="boxes",
        dtype="FLOAT32",
        shape=[1, 5, 4],
        shape_signature=[1, 5, 4],
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
    )
    model_ir.tensors["max_output"] = TensorIR(
        name="max_output",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([3], dtype=np.int32),
    )
    model_ir.tensors["iou_threshold"] = TensorIR(
        name="iou_threshold",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
    )
    model_ir.tensors["score_threshold"] = TensorIR(
        name="score_threshold",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.4], dtype=np.float32),
    )
    model_ir.tensors["selected_indices"] = TensorIR(
        name="selected_indices",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
    )
    model_ir.tensors["valid_count"] = TensorIR(
        name="valid_count",
        dtype="INT32",
        shape=[],
        shape_signature=[],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="NON_MAX_SUPPRESSION_V4",
            inputs=["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
            outputs=["selected_indices", "valid_count"],
            options={},
        )
    )
    return model_ir


def _make_shape_derived_non_max_suppression_v4_model_ir() -> ModelIR:
    model_ir = ModelIR(name="shape_derived_non_max_suppression_v4_model_ir")
    model_ir.inputs = ["boxes", "scores"]
    model_ir.outputs = ["selected_indices", "valid_count"]
    model_ir.tensors["boxes"] = TensorIR(
        name="boxes",
        dtype="FLOAT32",
        shape=[5, 4],
        shape_signature=[5, 4],
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[5],
        shape_signature=[5],
    )
    model_ir.tensors["boxes_shape"] = TensorIR(
        name="boxes_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
    )
    model_ir.tensors["num_boxes_index"] = TensorIR(
        name="num_boxes_index",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["num_boxes"] = TensorIR(
        name="num_boxes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["max_cap"] = TensorIR(
        name="max_cap",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2147483647], dtype=np.int32),
    )
    model_ir.tensors["max_output"] = TensorIR(
        name="max_output",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["iou_threshold"] = TensorIR(
        name="iou_threshold",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
    )
    model_ir.tensors["score_threshold"] = TensorIR(
        name="score_threshold",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.4], dtype=np.float32),
    )
    model_ir.tensors["selected_indices"] = TensorIR(
        name="selected_indices",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
    )
    model_ir.tensors["valid_count"] = TensorIR(
        name="valid_count",
        dtype="INT32",
        shape=[],
        shape_signature=[],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="SHAPE",
                inputs=["boxes"],
                outputs=["boxes_shape"],
                options={"outType": "INT32"},
            ),
            OperatorIR(
                op_type="GATHER",
                inputs=["boxes_shape", "num_boxes_index"],
                outputs=["num_boxes"],
                options={"axis": 0, "batchDims": 0},
            ),
            OperatorIR(
                op_type="MINIMUM",
                inputs=["max_cap", "num_boxes"],
                outputs=["max_output"],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="NON_MAX_SUPPRESSION_V4",
                inputs=["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
                outputs=["selected_indices", "valid_count"],
                options={},
            ),
        ]
    )
    return model_ir


def _make_dynamic_shape_derived_non_max_suppression_v4_model_ir() -> ModelIR:
    model_ir = _make_shape_derived_non_max_suppression_v4_model_ir()
    model_ir.name = "dynamic_shape_derived_non_max_suppression_v4_model_ir"
    model_ir.tensors["boxes"].shape_signature = [-1, 4]
    model_ir.tensors["scores"].shape_signature = [-1]
    model_ir.tensors["selected_indices"].shape_signature = [-1]
    return model_ir


def _make_nms_postprocess_chain_model_ir(*, score_threshold: float) -> ModelIR:
    model_ir = ModelIR(name="nms_postprocess_chain_model_ir")
    model_ir.inputs = ["boxes", "scores"]
    model_ir.outputs = ["selected_indices", "valid_count", "head"]
    model_ir.tensors["boxes"] = TensorIR(
        name="boxes",
        dtype="FLOAT32",
        shape=[1, 5, 4],
        shape_signature=[1, 5, 4],
        logical_layout="NCW",
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
        logical_layout="NCW",
    )
    model_ir.tensors["max_output"] = TensorIR(
        name="max_output",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([3], dtype=np.int32),
    )
    model_ir.tensors["iou_threshold"] = TensorIR(
        name="iou_threshold",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
    )
    model_ir.tensors["score_threshold"] = TensorIR(
        name="score_threshold",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([score_threshold], dtype=np.float32),
    )
    model_ir.tensors["selected_indices"] = TensorIR(
        name="selected_indices",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
    )
    model_ir.tensors["valid_count"] = TensorIR(
        name="valid_count",
        dtype="INT32",
        shape=[],
        shape_signature=[],
    )
    model_ir.tensors["selected_indices_i64"] = TensorIR(
        name="selected_indices_i64",
        dtype="INT64",
        shape=[3],
        shape_signature=[-1],
    )
    model_ir.tensors["gathered_boxes"] = TensorIR(
        name="gathered_boxes",
        dtype="FLOAT32",
        shape=[1, 3, 4],
        shape_signature=[1, -1, 4],
        logical_layout="NCW",
    )
    model_ir.tensors["score_map_shape"] = TensorIR(
        name="score_map_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 5], dtype=np.int32),
    )
    model_ir.tensors["score_map"] = TensorIR(
        name="score_map",
        dtype="FLOAT32",
        shape=[1, 5],
        shape_signature=[1, 5],
    )
    model_ir.tensors["selected_indices_2d_shape"] = TensorIR(
        name="selected_indices_2d_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([-1, 1], dtype=np.int32),
    )
    model_ir.tensors["selected_indices_2d"] = TensorIR(
        name="selected_indices_2d",
        dtype="INT64",
        shape=[3, 1],
        shape_signature=[-1, 1],
    )
    model_ir.tensors["batch_zeros"] = TensorIR(
        name="batch_zeros",
        dtype="INT64",
        shape=[3, 1],
        shape_signature=[-1, 1],
    )
    model_ir.tensors["gather_coords"] = TensorIR(
        name="gather_coords",
        dtype="INT64",
        shape=[3, 2],
        shape_signature=[-1, 2],
    )
    model_ir.tensors["gathered_scores"] = TensorIR(
        name="gathered_scores",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[-1],
    )
    model_ir.tensors["boxes_slice_begin"] = TensorIR(
        name="boxes_slice_begin",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 0, 0], dtype=np.int32),
    )
    model_ir.tensors["boxes_slice_size"] = TensorIR(
        name="boxes_slice_size",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, -1, 4], dtype=np.int32),
    )
    model_ir.tensors["sliced_boxes"] = TensorIR(
        name="sliced_boxes",
        dtype="FLOAT32",
        shape=[1, 3, 4],
        shape_signature=[1, -1, 4],
        logical_layout="NCW",
    )
    model_ir.tensors["gathered_scores_3d_shape"] = TensorIR(
        name="gathered_scores_3d_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, -1, 1], dtype=np.int32),
    )
    model_ir.tensors["gathered_scores_3d"] = TensorIR(
        name="gathered_scores_3d",
        dtype="FLOAT32",
        shape=[1, 3, 1],
        shape_signature=[1, -1, 1],
        logical_layout="NCW",
    )
    model_ir.tensors["head"] = TensorIR(
        name="head",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, -1, 5],
        logical_layout="NCW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="NON_MAX_SUPPRESSION_V4",
                inputs=["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
                outputs=["selected_indices", "valid_count"],
                options={},
            ),
            OperatorIR(
                op_type="CAST",
                inputs=["selected_indices"],
                outputs=["selected_indices_i64"],
                options={"outDataType": "INT64"},
            ),
            OperatorIR(
                op_type="GATHER",
                inputs=["boxes", "selected_indices_i64"],
                outputs=["gathered_boxes"],
                options={"axis": 1, "batchDims": 0},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["scores", "score_map_shape"],
                outputs=["score_map"],
                options={},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["selected_indices_i64", "selected_indices_2d_shape"],
                outputs=["selected_indices_2d"],
                options={},
            ),
            OperatorIR(
                op_type="SUB",
                inputs=["selected_indices_2d", "selected_indices_2d"],
                outputs=["batch_zeros"],
                options={},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["batch_zeros", "selected_indices_2d"],
                outputs=["gather_coords"],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="GATHER_ND",
                inputs=["score_map", "gather_coords"],
                outputs=["gathered_scores"],
                options={},
            ),
            OperatorIR(
                op_type="SLICE",
                inputs=["gathered_boxes", "boxes_slice_begin", "boxes_slice_size"],
                outputs=["sliced_boxes"],
                options={},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["gathered_scores", "gathered_scores_3d_shape"],
                outputs=["gathered_scores_3d"],
                options={},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["sliced_boxes", "gathered_scores_3d"],
                outputs=["head"],
                options={"axis": 2, "fusedActivationFunction": "NONE"},
            ),
        ]
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


def _make_concat_with_layout_alignment_model_ir() -> ModelIR:
    model_ir = ModelIR(name="concat_with_layout_alignment_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 4, 3],
        shape_signature=[1, 1, 4, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 4, 3],
        shape_signature=[1, 1, 4, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 3, 2, 4],
        shape_signature=[1, 3, 2, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x", "y"],
            outputs=["z"],
            options={"axis": 2, "fusedActivationFunction": "NONE"},
        )
    )
    return model_ir


def _make_concat_with_ambiguous_axis_only_match_model_ir() -> ModelIR:
    model_ir = ModelIR(name="concat_with_ambiguous_axis_only_match_model_ir")
    model_ir.inputs = ["boxes", "obj", "cls"]
    model_ir.outputs = ["head"]
    model_ir.tensors["boxes"] = TensorIR(
        name="boxes",
        dtype="FLOAT32",
        shape=[1, 4, 8, 8],
        shape_signature=[1, 4, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["obj"] = TensorIR(
        name="obj",
        dtype="FLOAT32",
        shape=[1, 1, 8, 8],
        shape_signature=[1, 1, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["cls"] = TensorIR(
        name="cls",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["head"] = TensorIR(
        name="head",
        dtype="FLOAT32",
        shape=[1, 13, 8, 8],
        shape_signature=[1, 13, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["boxes", "obj", "cls"],
            outputs=["head"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        )
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


def _make_dynamic_binary_scalar_rhs_model_ir() -> ModelIR:
    model_ir = ModelIR(name="dynamic_binary_scalar_rhs_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y_add", "y_less"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["zero"] = TensorIR(
        name="zero",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["offset"] = TensorIR(
        name="offset",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([19248], dtype=np.int32),
    )
    model_ir.tensors["y_less"] = TensorIR(
        name="y_less",
        dtype="BOOL",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["y_add"] = TensorIR(
        name="y_add",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="LESS", inputs=["x", "zero"], outputs=["y_less"], options={}),
            OperatorIR(op_type="ADD", inputs=["x", "offset"], outputs=["y_add"], options={}),
        ]
    )
    return model_ir


def _make_concat_with_rank3_dynamic_middle_dim_model_ir() -> ModelIR:
    model_ir = ModelIR(name="concat_with_rank3_dynamic_middle_dim_model_ir")
    model_ir.inputs = ["boxes", "scores", "classes"]
    model_ir.outputs = ["head"]
    model_ir.tensors["boxes"] = TensorIR(
        name="boxes",
        dtype="FLOAT32",
        shape=[1, 34, 4],
        shape_signature=[1, 34, 4],
        logical_layout="NCW",
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1, 34, 1],
        shape_signature=[1, 34, 1],
        logical_layout="NCW",
    )
    model_ir.tensors["classes"] = TensorIR(
        name="classes",
        dtype="FLOAT32",
        shape=[1, 34, 1],
        shape_signature=[1, 34, 1],
        logical_layout="NCW",
    )
    model_ir.tensors["head"] = TensorIR(
        name="head",
        dtype="FLOAT32",
        shape=[1, 1, 6],
        shape_signature=[1, -1, 6],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["boxes", "scores", "classes"],
            outputs=["head"],
            options={"axis": 2, "fusedActivationFunction": "NONE"},
        )
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


def _make_static_sigmoid_while_model_ir() -> ModelIR:
    model_ir = ModelIR(name="static_sigmoid_while")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
    )
    model_ir.tensors["iter_init"] = TensorIR(
        name="iter_init",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int64),
    )
    model_ir.tensors["trip_count"] = TensorIR(
        name="trip_count",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([8], dtype=np.int64),
    )
    model_ir.tensors["cond_init"] = TensorIR(
        name="cond_init",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([True], dtype=bool),
    )
    for name, dtype in [
        ("iter_out", "INT64"),
        ("trip_out", "INT64"),
        ("cond_out", "BOOL"),
    ]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[1],
            shape_signature=[1],
        )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
    )

    cond_subgraph = ModelIR(name="static_sigmoid_while_cond")
    cond_subgraph.inputs = ["iter_in", "trip_in", "cond_in", "state_in"]
    cond_subgraph.outputs = ["cond_eval"]
    cond_subgraph.tensors["iter_in"] = TensorIR(name="iter_in", dtype="INT64", shape=[1], shape_signature=[1])
    cond_subgraph.tensors["trip_in"] = TensorIR(name="trip_in", dtype="INT64", shape=[1], shape_signature=[1])
    cond_subgraph.tensors["cond_in"] = TensorIR(name="cond_in", dtype="BOOL", shape=[1], shape_signature=[1])
    cond_subgraph.tensors["state_in"] = TensorIR(name="state_in", dtype="FLOAT32", shape=[3], shape_signature=[3])
    cond_subgraph.tensors["iter_lt_trip"] = TensorIR(
        name="iter_lt_trip",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
    )
    cond_subgraph.tensors["cond_eval"] = TensorIR(
        name="cond_eval",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
    )
    cond_subgraph.operators.extend(
        [
            OperatorIR(
                op_type="LESS",
                inputs=["iter_in", "trip_in"],
                outputs=["iter_lt_trip"],
                options={},
            ),
            OperatorIR(
                op_type="LOGICAL_AND",
                inputs=["cond_in", "iter_lt_trip"],
                outputs=["cond_eval"],
                options={},
            ),
        ]
    )

    body_subgraph = ModelIR(name="static_sigmoid_while_body")
    body_subgraph.inputs = ["iter_in", "trip_in", "cond_in", "state_in"]
    body_subgraph.outputs = ["iter_out_body", "trip_out_body", "cond_out_body", "state_out_body"]
    body_subgraph.tensors["iter_in"] = TensorIR(name="iter_in", dtype="INT64", shape=[1], shape_signature=[1])
    body_subgraph.tensors["trip_in"] = TensorIR(name="trip_in", dtype="INT64", shape=[1], shape_signature=[1])
    body_subgraph.tensors["cond_in"] = TensorIR(name="cond_in", dtype="BOOL", shape=[1], shape_signature=[1])
    body_subgraph.tensors["state_in"] = TensorIR(name="state_in", dtype="FLOAT32", shape=[3], shape_signature=[3])
    body_subgraph.tensors["iter_plus_one_const"] = TensorIR(
        name="iter_plus_one_const",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int64),
    )
    body_subgraph.tensors["trip_shape"] = TensorIR(
        name="trip_shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    body_subgraph.tensors["cond_shape"] = TensorIR(
        name="cond_shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    body_subgraph.tensors["state_shape"] = TensorIR(
        name="state_shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([3], dtype=np.int32),
    )
    body_subgraph.tensors["iter_out_body"] = TensorIR(
        name="iter_out_body",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
    )
    body_subgraph.tensors["trip_out_body"] = TensorIR(
        name="trip_out_body",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
    )
    body_subgraph.tensors["cond_out_body"] = TensorIR(
        name="cond_out_body",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
    )
    previous_name = "state_in"
    for idx in range(8):
        output_name = "state_out_raw" if idx == 7 else f"sigmoid_{idx}"
        body_subgraph.tensors[output_name] = TensorIR(
            name=output_name,
            dtype="FLOAT32",
            shape=[3],
            shape_signature=[3],
        )
        body_subgraph.operators.append(
            OperatorIR(
                op_type="LOGISTIC",
                inputs=[previous_name],
                outputs=[output_name],
                options={},
            )
        )
        previous_name = output_name
    body_subgraph.tensors["state_out_body"] = TensorIR(
        name="state_out_body",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
    )
    body_subgraph.operators[0:0] = [
        OperatorIR(
            op_type="ADD",
            inputs=["iter_in", "iter_plus_one_const"],
            outputs=["iter_out_body"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["trip_in", "trip_shape"],
            outputs=["trip_out_body"],
            options={"newShape": [1], "allowZero": False},
        ),
    ]
    body_subgraph.operators.extend(
        [
            OperatorIR(
                op_type="RESHAPE",
                inputs=["cond_in", "cond_shape"],
                outputs=["cond_out_body"],
                options={"newShape": [1], "allowZero": False},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["state_out_raw", "state_shape"],
                outputs=["state_out_body"],
                options={"newShape": [3], "allowZero": False},
            ),
        ]
    )

    model_ir.subgraphs = [cond_subgraph, body_subgraph]
    model_ir.operators.append(
        OperatorIR(
            op_type="WHILE",
            inputs=["iter_init", "trip_count", "cond_init", "x"],
            outputs=["iter_out", "trip_out", "cond_out", "y"],
            options={"condSubgraphIndex": 1, "bodySubgraphIndex": 2},
        )
    )
    return model_ir


def _make_counter_bounded_sigmoid_while_model_ir() -> ModelIR:
    model_ir = ModelIR(name="counter_bounded_sigmoid_while")
    model_ir.inputs = ["x", "counter"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
    )
    model_ir.tensors["counter"] = TensorIR(
        name="counter",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["counter_i32"] = TensorIR(
        name="counter_i32",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["threshold_i32"] = TensorIR(
        name="threshold_i32",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([64], dtype=np.int32),
    )
    model_ir.tensors["cond_init"] = TensorIR(
        name="cond_init",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["iter_init"] = TensorIR(
        name="iter_init",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int64),
    )
    model_ir.tensors["trip_count"] = TensorIR(
        name="trip_count",
        dtype="INT64",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([9223372036854775807], dtype=np.int64),
    )
    for name, dtype in [
        ("iter_out", "INT64"),
        ("trip_out", "INT64"),
        ("cond_out", "BOOL"),
        ("counter_out", "INT64"),
    ]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[1],
            shape_signature=[1],
        )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="CAST",
                inputs=["counter"],
                outputs=["counter_i32"],
                options={"inDataType": "INT64", "outDataType": "INT32"},
            ),
            OperatorIR(
                op_type="LESS",
                inputs=["counter_i32", "threshold_i32"],
                outputs=["cond_init"],
                options={},
            ),
        ]
    )

    cond_subgraph = ModelIR(name="counter_bounded_sigmoid_while_cond")
    cond_subgraph.inputs = ["iter_in", "trip_in", "cond_in", "state_in", "counter_in"]
    cond_subgraph.outputs = ["cond_eval"]
    for name, dtype, shape in [
        ("iter_in", "INT64", [1]),
        ("trip_in", "INT64", [1]),
        ("cond_in", "BOOL", [1]),
        ("state_in", "FLOAT32", [3]),
        ("counter_in", "INT64", [1]),
        ("iter_lt_trip", "BOOL", [1]),
        ("cond_eval", "BOOL", [1]),
    ]:
        cond_subgraph.tensors[name] = TensorIR(name=name, dtype=dtype, shape=shape, shape_signature=shape)
    cond_subgraph.operators.extend(
        [
            OperatorIR(op_type="LESS", inputs=["iter_in", "trip_in"], outputs=["iter_lt_trip"], options={}),
            OperatorIR(op_type="LOGICAL_AND", inputs=["cond_in", "iter_lt_trip"], outputs=["cond_eval"], options={}),
        ]
    )

    body_subgraph = ModelIR(name="counter_bounded_sigmoid_while_body")
    body_subgraph.inputs = ["iter_in", "trip_in", "cond_in", "state_in", "counter_in"]
    body_subgraph.outputs = ["iter_out_body", "trip_out_body", "cond_out_body", "state_out_body", "counter_out_body"]
    tensor_specs = [
        ("iter_in", "INT64", [1], None),
        ("trip_in", "INT64", [1], None),
        ("cond_in", "BOOL", [1], None),
        ("state_in", "FLOAT32", [3], None),
        ("counter_in", "INT64", [1], None),
        ("iter_plus_one_const", "INT64", [1], np.asarray([1], dtype=np.int64)),
        ("counter_plus_one_const", "INT64", [1], np.asarray([1], dtype=np.int64)),
        ("trip_shape", "INT32", [1], np.asarray([1], dtype=np.int32)),
        ("state_shape", "INT32", [1], np.asarray([3], dtype=np.int32)),
        ("counter_shape", "INT32", [1], np.asarray([1], dtype=np.int32)),
        ("threshold_i64", "INT64", [1], np.asarray([64], dtype=np.int64)),
        ("threshold_i32", "INT32", [1], np.asarray([64], dtype=np.int32)),
        ("iter_out_body", "INT64", [1], None),
        ("trip_out_body", "INT64", [1], None),
        ("state_out_raw", "FLOAT32", [3], None),
        ("state_out_body", "FLOAT32", [3], None),
        ("counter_out_raw", "INT64", [1], None),
        ("counter_out_body", "INT64", [1], None),
        ("counter_out_i32", "INT32", [1], None),
        ("cond_out_body", "BOOL", [1], None),
    ]
    for name, dtype, shape, data in tensor_specs:
        body_subgraph.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=shape,
            shape_signature=shape,
            data=data,
        )
    body_subgraph.operators.extend(
        [
            OperatorIR(op_type="ADD", inputs=["iter_in", "iter_plus_one_const"], outputs=["iter_out_body"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="RESHAPE", inputs=["trip_in", "trip_shape"], outputs=["trip_out_body"], options={"newShape": [1], "allowZero": False}),
            OperatorIR(op_type="LOGISTIC", inputs=["state_in"], outputs=["state_out_raw"], options={}),
            OperatorIR(op_type="ADD", inputs=["counter_in", "counter_plus_one_const"], outputs=["counter_out_raw"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="CAST", inputs=["counter_out_raw"], outputs=["counter_out_i32"], options={"inDataType": "INT64", "outDataType": "INT32"}),
            OperatorIR(op_type="CAST", inputs=["threshold_i64"], outputs=["threshold_i32"], options={"inDataType": "INT64", "outDataType": "INT32"}),
            OperatorIR(op_type="LESS", inputs=["counter_out_i32", "threshold_i32"], outputs=["cond_out_body"], options={}),
            OperatorIR(op_type="RESHAPE", inputs=["state_out_raw", "state_shape"], outputs=["state_out_body"], options={"newShape": [3], "allowZero": False}),
            OperatorIR(op_type="RESHAPE", inputs=["counter_out_raw", "counter_shape"], outputs=["counter_out_body"], options={"newShape": [1], "allowZero": False}),
        ]
    )

    model_ir.subgraphs = [cond_subgraph, body_subgraph]
    model_ir.operators.append(
        OperatorIR(
            op_type="WHILE",
            inputs=["iter_init", "trip_count", "cond_init", "x", "counter"],
            outputs=["iter_out", "trip_out", "cond_out", "y", "counter_out"],
            options={"condSubgraphIndex": 1, "bodySubgraphIndex": 2},
        )
    )
    return model_ir


def _make_counter_bounded_select_sigmoid_while_model_ir() -> ModelIR:
    model_ir = _make_counter_bounded_sigmoid_while_model_ir()
    model_ir.name = "counter_bounded_select_sigmoid_while"
    body_subgraph = model_ir.subgraphs[1]
    body_subgraph.name = "counter_bounded_select_sigmoid_while_body"
    body_subgraph.tensors["cond_false"] = TensorIR(
        name="cond_false",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([False], dtype=bool),
    )
    body_subgraph.tensors["cond_true"] = TensorIR(
        name="cond_true",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([True], dtype=bool),
    )
    body_subgraph.tensors["counter_ge_threshold"] = TensorIR(
        name="counter_ge_threshold",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
    )
    less_index = next(
        index
        for index, op in enumerate(body_subgraph.operators)
        if op.op_type == "LESS" and list(op.outputs) == ["cond_out_body"]
    )
    body_subgraph.operators[less_index] = OperatorIR(
        op_type="GREATER_EQUAL",
        inputs=["counter_out_i32", "threshold_i32"],
        outputs=["counter_ge_threshold"],
        options={},
    )
    body_subgraph.operators.insert(
        less_index + 1,
        OperatorIR(
            op_type="SELECT",
            inputs=["counter_ge_threshold", "cond_false", "cond_true"],
            outputs=["cond_out_body"],
            options={},
        ),
    )
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


def _make_runtime_wrapper_pool_nhwc_model_ir() -> ModelIR:
    model_ir = ModelIR(name="runtime_wrapper_pool_nhwc_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 6, 2],
        shape_signature=[1, 4, 6, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 6, 2],
        shape_signature=[1, 4, 6, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 3, 2],
        shape_signature=[1, 2, 3, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 2],
        shape_signature=[1, 2, 3, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["zero"] = TensorIR(
        name="zero",
        dtype="FLOAT32",
        shape=[1, 4, 6, 2],
        shape_signature=[1, 4, 6, 2],
        data=np.zeros((1, 4, 6, 2), dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["zero_out"] = TensorIR(
        name="zero_out",
        dtype="FLOAT32",
        shape=[1, 2, 3, 2],
        shape_signature=[1, 2, 3, 2],
        data=np.zeros((1, 2, 3, 2), dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="ADD", inputs=["x", "zero"], outputs=["x_nhwc"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(
                op_type="MAX_POOL_2D",
                inputs=["x_nhwc"],
                outputs=["y_nhwc"],
                options={
                    "padding": "VALID",
                    "strideW": 2,
                    "strideH": 2,
                    "filterWidth": 2,
                    "filterHeight": 2,
                    "fusedActivationFunction": "NONE",
                },
            ),
            OperatorIR(op_type="ADD", inputs=["y_nhwc", "zero_out"], outputs=["y"], options={"fusedActivationFunction": "NONE"}),
        ]
    )
    return model_ir


def _make_nested_static_reshape_model_ir() -> ModelIR:
    model_ir = ModelIR(name="nested_static_reshape_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 5],
        shape_signature=[1, 3, 4, 5],
        logical_layout="NCHW",
    )
    model_ir.tensors["mid"] = TensorIR(
        name="mid",
        dtype="FLOAT32",
        shape=[1, 3, 20],
        shape_signature=[1, 3, 20],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape_mid"] = TensorIR(
        name="shape_mid",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 3, 20], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 60],
        shape_signature=[1, 60],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape_out"] = TensorIR(
        name="shape_out",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 60], dtype=np.int32),
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="RESHAPE", inputs=["x", "shape_mid"], outputs=["mid"], options={"newShape": [1, 3, 20]}),
            OperatorIR(op_type="RESHAPE", inputs=["mid", "shape_out"], outputs=["y"], options={"newShape": [1, 60]}),
        ]
    )
    return model_ir


def _make_same_shape_max_pool_codegen_model_ir() -> ModelIR:
    model_ir = ModelIR(name="same_shape_max_pool_codegen_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 256, 15, 20],
        shape_signature=[1, 256, 15, 20],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 256, 15, 20],
        shape_signature=[1, 256, 15, 20],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="MAX_POOL_2D",
            inputs=["x"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideW": 1,
                "strideH": 1,
                "filterWidth": 5,
                "filterHeight": 5,
                "fusedActivationFunction": "NONE",
            },
        )
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


def _make_conv2d_same_stride2_relu_model_ir() -> ModelIR:
    rng = np.random.default_rng(41)
    model_ir = ModelIR(name="conv2d_same_stride2_relu_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
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
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 2, 2],
        shape_signature=[1, 2, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 2,
                "strideW": 2,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "RELU",
            },
        )
    )
    return model_ir


def _make_conv2d_permuted_hwc_input_model_ir() -> ModelIR:
    rng = np.random.default_rng(123)
    model_ir = ModelIR(name="conv2d_permuted_hwc_input_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 86, 2, 51],
        shape_signature=[1, 86, 2, 51],
        logical_layout="NCHW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 7, 7, 2],
        shape_signature=[1, 7, 7, 2],
        data=rng.standard_normal((1, 7, 7, 2)).astype(np.float32),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=rng.standard_normal((1,)).astype(np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 45, 80],
        shape_signature=[1, 1, 45, 80],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "padding": "VALID",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        )
    )
    return model_ir


def _make_nhwc_same_conv_concat_model_ir() -> ModelIR:
    rng = np.random.default_rng(321)
    model_ir = ModelIR(name="nhwc_same_conv_concat_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 40, 120],
        shape_signature=[1, 1, 40, 120],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[512, 1, 1, 120],
        shape_signature=[512, 1, 1, 120],
        data=rng.standard_normal((512, 1, 1, 120)).astype(np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[512],
        shape_signature=[512],
        data=rng.standard_normal((512,)).astype(np.float32),
    )
    model_ir.tensors["conv_out"] = TensorIR(
        name="conv_out",
        dtype="FLOAT32",
        shape=[1, 1, 40, 512],
        shape_signature=[1, 1, 40, 512],
        logical_layout="NHWC",
    )
    model_ir.tensors["skip"] = TensorIR(
        name="skip",
        dtype="FLOAT32",
        shape=[1, 1, 40, 512],
        shape_signature=[1, 1, 40, 512],
        data=rng.standard_normal((1, 1, 40, 512)).astype(np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 40, 1024],
        shape_signature=[1, 1, 40, 1024],
        logical_layout="NHWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["conv_out"],
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
    model_ir.operators.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["skip", "conv_out"],
            outputs=["y"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        )
    )
    return model_ir


def _make_nhwc_same_depthwise_conv_model_ir() -> ModelIR:
    rng = np.random.default_rng(654)
    model_ir = ModelIR(name="nhwc_same_depthwise_conv_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 6, 160, 256],
        shape_signature=[1, 6, 160, 256],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 5, 5, 256],
        shape_signature=[1, 5, 5, 256],
        data=rng.standard_normal((1, 5, 5, 256)).astype(np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[256],
        shape_signature=[256],
        data=rng.standard_normal((256,)).astype(np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 6, 160, 256],
        shape_signature=[1, 6, 160, 256],
        logical_layout="NHWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="DEPTHWISE_CONV_2D",
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



def _make_mirror_pad_with_noop_outer_dims_model_ir() -> ModelIR:
    model_ir = ModelIR(name="mirror_pad_with_noop_outer_dims_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 4, 4],
        shape_signature=[1, 2, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["pads"] = TensorIR(
        name="pads",
        dtype="INT32",
        shape=[4, 2],
        shape_signature=[4, 2],
        data=np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 6, 6],
        shape_signature=[1, 2, 6, 6],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="MIRROR_PAD",
            inputs=["x", "pads"],
            outputs=["y"],
            options={},
        )
    )
    return model_ir


def _make_mirror_pad_with_non_suffix_axes_model_ir() -> ModelIR:
    model_ir = ModelIR(name="mirror_pad_with_non_suffix_axes_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 2, 5],
        shape_signature=[1, 4, 2, 5],
        logical_layout="NCHW",
    )
    model_ir.tensors["pads"] = TensorIR(
        name="pads",
        dtype="INT32",
        shape=[4, 2],
        shape_signature=[4, 2],
        data=np.asarray([[0, 0], [1, 1], [0, 0], [2, 2]], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 6, 2, 9],
        shape_signature=[1, 6, 2, 9],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="MIRROR_PAD",
            inputs=["x", "pads"],
            outputs=["y"],
            options={},
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


def _make_boundary_wrapped_conv_relu_conv_model_ir() -> ModelIR:
    model_ir = ModelIR(name="boundary_wrapped_conv_relu_conv")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [1, 3, 8, 8],
        "y": [1, 5, 8, 8],
    }
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 8, 8],
        shape_signature=[1, 3, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_nhwc"] = TensorIR(
        name="perm_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 3],
        shape_signature=[1, 8, 8, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["w0"] = TensorIR(
        name="w0",
        dtype="FLOAT32",
        shape=[4, 3, 3, 3],
        shape_signature=[4, 3, 3, 3],
        data=np.zeros((4, 3, 3, 3), dtype=np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b0"] = TensorIR(
        name="b0",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=np.zeros((4,), dtype=np.float32),
    )
    model_ir.tensors["conv0_nhwc"] = TensorIR(
        name="conv0_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["relu0_nhwc"] = TensorIR(
        name="relu0_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["w1"] = TensorIR(
        name="w1",
        dtype="FLOAT32",
        shape=[5, 4, 3, 3],
        shape_signature=[5, 4, 3, 3],
        data=np.zeros((5, 4, 3, 3), dtype=np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b1"] = TensorIR(
        name="b1",
        dtype="FLOAT32",
        shape=[5],
        shape_signature=[5],
        data=np.zeros((5,), dtype=np.float32),
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 5],
        shape_signature=[1, 8, 8, 5],
        logical_layout="NHWC",
    )
    model_ir.tensors["perm_nchw"] = TensorIR(
        name="perm_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 8, 8],
        shape_signature=[1, 5, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_nhwc"], outputs=["x_nhwc"], options={}),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_nhwc", "w0", "b0"],
                outputs=["conv0_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="RELU", inputs=["conv0_nhwc"], outputs=["relu0_nhwc"], options={}),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["relu0_nhwc", "w1", "b1"],
                outputs=["y_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="TRANSPOSE", inputs=["y_nhwc", "perm_nchw"], outputs=["y"], options={}),
        ]
    )
    return model_ir


def _make_boundary_wrapped_conv_pool_conv_model_ir() -> ModelIR:
    model_ir = ModelIR(name="boundary_wrapped_conv_pool_conv")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [1, 3, 8, 8],
        "y": [1, 5, 4, 4],
    }
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 8, 8],
        shape_signature=[1, 3, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_nhwc"] = TensorIR(
        name="perm_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 3],
        shape_signature=[1, 8, 8, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["w0"] = TensorIR(
        name="w0",
        dtype="FLOAT32",
        shape=[4, 3, 3, 3],
        shape_signature=[4, 3, 3, 3],
        data=np.zeros((4, 3, 3, 3), dtype=np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b0"] = TensorIR(
        name="b0",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=np.zeros((4,), dtype=np.float32),
    )
    model_ir.tensors["conv0_nhwc"] = TensorIR(
        name="conv0_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["pool0_nhwc"] = TensorIR(
        name="pool0_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 4],
        shape_signature=[1, 4, 4, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["w1"] = TensorIR(
        name="w1",
        dtype="FLOAT32",
        shape=[5, 4, 3, 3],
        shape_signature=[5, 4, 3, 3],
        data=np.zeros((5, 4, 3, 3), dtype=np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b1"] = TensorIR(
        name="b1",
        dtype="FLOAT32",
        shape=[5],
        shape_signature=[5],
        data=np.zeros((5,), dtype=np.float32),
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 5],
        shape_signature=[1, 4, 4, 5],
        logical_layout="NHWC",
    )
    model_ir.tensors["perm_nchw"] = TensorIR(
        name="perm_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 4, 4],
        shape_signature=[1, 5, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_nhwc"], outputs=["x_nhwc"], options={}),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_nhwc", "w0", "b0"],
                outputs=["conv0_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="MAX_POOL_2D",
                inputs=["conv0_nhwc"],
                outputs=["pool0_nhwc"],
                options={"filterHeight": 2, "filterWidth": 2, "strideH": 2, "strideW": 2, "padding": "VALID", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["pool0_nhwc", "w1", "b1"],
                outputs=["y_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="TRANSPOSE", inputs=["y_nhwc", "perm_nchw"], outputs=["y"], options={}),
        ]
    )
    return model_ir


def _make_boundary_wrapped_conv_add_conv_model_ir() -> ModelIR:
    model_ir = ModelIR(name="boundary_wrapped_conv_add_conv")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [1, 3, 8, 8],
        "y": [1, 6, 8, 8],
    }
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 8, 8],
        shape_signature=[1, 3, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_nhwc"] = TensorIR(
        name="perm_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["perm_nchw"] = TensorIR(
        name="perm_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 3],
        shape_signature=[1, 8, 8, 3],
        logical_layout="NHWC",
    )
    for name, out_channels, in_channels, scale in (
        ("w0", 4, 3, 0.05),
        ("w1", 4, 4, 0.04),
        ("w2", 6, 4, 0.03),
    ):
        count = out_channels * in_channels * 3 * 3
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[out_channels, in_channels, 3, 3],
            shape_signature=[out_channels, in_channels, 3, 3],
            data=(np.arange(count, dtype=np.float32).reshape(out_channels, in_channels, 3, 3) * scale),
        )
    for name, values in (
        ("b0", np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)),
        ("b1", np.asarray([0.2, -0.1, 0.05, -0.15], dtype=np.float32)),
        ("b2", np.asarray([0.1, -0.05, 0.2, -0.1, 0.15, -0.2], dtype=np.float32)),
    ):
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[len(values)],
            shape_signature=[len(values)],
            data=values,
        )
    model_ir.tensors["conv0_nhwc"] = TensorIR(
        name="conv0_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["conv1_nhwc"] = TensorIR(
        name="conv1_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["add0_nhwc"] = TensorIR(
        name="add0_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 6],
        shape_signature=[1, 8, 8, 6],
        logical_layout="NHWC",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 6, 8, 8],
        shape_signature=[1, 6, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_nhwc"], outputs=["x_nhwc"], options={}),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_nhwc", "w0", "b0"],
                outputs=["conv0_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["conv0_nhwc", "w1", "b1"],
                outputs=["conv1_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="ADD", inputs=["conv1_nhwc", "conv0_nhwc"], outputs=["add0_nhwc"], options={}),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["add0_nhwc", "w2", "b2"],
                outputs=["y_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="TRANSPOSE", inputs=["y_nhwc", "perm_nchw"], outputs=["y"], options={}),
        ]
    )
    return model_ir


def _make_boundary_wrapped_conv_dual_pool_concat_conv_model_ir() -> ModelIR:
    model_ir = ModelIR(name="boundary_wrapped_conv_dual_pool_concat_conv")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [1, 3, 8, 8],
        "y": [1, 5, 4, 4],
    }
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 8, 8],
        shape_signature=[1, 3, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_nhwc"] = TensorIR(
        name="perm_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["perm_nchw"] = TensorIR(
        name="perm_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 3],
        shape_signature=[1, 8, 8, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["w0"] = TensorIR(
        name="w0",
        dtype="FLOAT32",
        shape=[4, 3, 3, 3],
        shape_signature=[4, 3, 3, 3],
        data=np.zeros((4, 3, 3, 3), dtype=np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b0"] = TensorIR(
        name="b0",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=np.zeros((4,), dtype=np.float32),
    )
    model_ir.tensors["conv0_nhwc"] = TensorIR(
        name="conv0_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["max0_nhwc"] = TensorIR(
        name="max0_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 4],
        shape_signature=[1, 4, 4, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["avg0_nhwc"] = TensorIR(
        name="avg0_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 4],
        shape_signature=[1, 4, 4, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["cat0_nhwc"] = TensorIR(
        name="cat0_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["w1"] = TensorIR(
        name="w1",
        dtype="FLOAT32",
        shape=[5, 8, 1, 1],
        shape_signature=[5, 8, 1, 1],
        data=np.zeros((5, 8, 1, 1), dtype=np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b1"] = TensorIR(
        name="b1",
        dtype="FLOAT32",
        shape=[5],
        shape_signature=[5],
        data=np.zeros((5,), dtype=np.float32),
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 5],
        shape_signature=[1, 4, 4, 5],
        logical_layout="NHWC",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 4, 4],
        shape_signature=[1, 5, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_nhwc"], outputs=["x_nhwc"], options={}),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_nhwc", "w0", "b0"],
                outputs=["conv0_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "SAME", "fusedActivationFunction": "RELU"},
            ),
            OperatorIR(
                op_type="MAX_POOL_2D",
                inputs=["conv0_nhwc"],
                outputs=["max0_nhwc"],
                options={"filterHeight": 2, "filterWidth": 2, "strideH": 2, "strideW": 2, "padding": "VALID", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="AVERAGE_POOL_2D",
                inputs=["conv0_nhwc"],
                outputs=["avg0_nhwc"],
                options={"filterHeight": 2, "filterWidth": 2, "strideH": 2, "strideW": 2, "padding": "VALID", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["max0_nhwc", "avg0_nhwc"],
                outputs=["cat0_nhwc"],
                options={"axis": 3, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["cat0_nhwc", "w1", "b1"],
                outputs=["y_nhwc"],
                options={"strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1, "padding": "VALID", "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="TRANSPOSE", inputs=["y_nhwc", "perm_nchw"], outputs=["y"], options={}),
        ]
    )
    return model_ir


def _make_boundary_wrapped_gather_branch_model_ir() -> ModelIR:
    model_ir = ModelIR(name="boundary_wrapped_gather_branch")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [1, 3, 4, 4],
        "y": [1, 2, 4, 4],
    }
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_nhwc"] = TensorIR(
        name="perm_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["gather_indices"] = TensorIR(
        name="gather_indices",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([0, 2], dtype=np.int32),
    )
    model_ir.tensors["gathered_nhwc"] = TensorIR(
        name="gathered_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
        logical_layout="NHWC",
    )
    model_ir.tensors["perm_nchw"] = TensorIR(
        name="perm_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 4, 4],
        shape_signature=[1, 2, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_nhwc"], outputs=["x_nhwc"], options={}),
            OperatorIR(op_type="GATHER", inputs=["x_nhwc", "gather_indices"], outputs=["gathered_nhwc"], options={"axis": 3}),
            OperatorIR(op_type="TRANSPOSE", inputs=["gathered_nhwc", "perm_nchw"], outputs=["y"], options={}),
        ]
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


def _make_non_layout_shape_transpose_model_ir() -> ModelIR:
    model_ir = ModelIR(name="non_layout_shape_transpose_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[24, 2, 1936],
        shape_signature=[24, 2, 1936],
        logical_layout="NCW",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 0, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[2, 24, 1936],
        shape_signature=[2, 24, 1936],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["y"], options={})
    )
    return model_ir


def _make_degenerate_unknown_layout_slice_model_ir() -> ModelIR:
    model_ir = ModelIR(name="degenerate_unknown_layout_slice_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 1, 2],
        shape_signature=[1, 1, -1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["begin"] = TensorIR(
        name="begin",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 0, 0, 0], dtype=np.int32),
    )
    model_ir.tensors["size"] = TensorIR(
        name="size",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, -1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[1, 1, -1, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(op_type="SLICE", inputs=["x", "begin", "size"], outputs=["y"], options={})
    )
    return model_ir


def _make_scatter_nd_indices_unknown_layout_model_ir() -> ModelIR:
    model_ir = ModelIR(name="scatter_nd_indices_unknown_layout_model_ir")
    model_ir.inputs = ["updates"]
    model_ir.outputs = ["y"]
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1, 2, 4],
        shape_signature=[1, 1, 2, 4],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["updates"] = TensorIR(
        name="updates",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[1, 1, 2, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 2, 3], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[1, 1, 2, 3],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=["indices", "updates", "shape"],
            outputs=["y"],
            options={},
        )
    )
    return model_ir


def _make_scatter_nd_updates_unknown_layout_model_ir() -> ModelIR:
    model_ir = ModelIR(name="scatter_nd_updates_unknown_layout_model_ir")
    model_ir.inputs = ["updates"]
    model_ir.outputs = ["y"]
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1, 2, 4],
        shape_signature=[1, 1, 2, 4],
        data=np.asarray([[[[0, 0, 0, 1], [0, 0, 1, 2]]]], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["updates"] = TensorIR(
        name="updates",
        dtype="FLOAT32",
        shape=[1, 1, 2],
        shape_signature=[1, 1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 2, 3], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[1, 1, 2, 3],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=["indices", "updates", "shape"],
            outputs=["y"],
            options={},
        )
    )
    return model_ir


def _make_scatter_nd_internal_output_unknown_layout_model_ir() -> ModelIR:
    model_ir = ModelIR(name="scatter_nd_internal_output_unknown_layout_model_ir")
    model_ir.inputs = ["data"]
    model_ir.outputs = ["y"]
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1, 2, 4],
        shape_signature=[1, 1, 2, 4],
        data=np.asarray([[[[0, 0, 0, 1], [0, 0, 1, 2]]]], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["updates"] = TensorIR(
        name="updates",
        dtype="FLOAT32",
        shape=[1, 1, 2],
        shape_signature=[1, 1, 2],
        data=np.asarray([[[3.0, 7.0]]], dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 2, 3], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["data"] = TensorIR(
        name="data",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[1, 1, 2, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["scattered"] = TensorIR(
        name="scattered",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[-1, -1, -1, -1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[1, 1, 2, 3],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=["indices", "updates", "shape"],
            outputs=["scattered"],
            options={},
        )
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ADD",
            inputs=["data", "scattered"],
            outputs=["y"],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    return model_ir


def _make_scatter_nd_channel_first_updates_layout_model_ir() -> ModelIR:
    model_ir = ModelIR(name="scatter_nd_channel_first_updates_layout_model_ir")
    model_ir.inputs = ["updates"]
    model_ir.outputs = ["y"]
    indices = np.zeros((1, 1, 2, 3, 4, 5), dtype=np.int32)
    for h in range(2):
        for w in range(3):
            for c in range(4):
                indices[0, 0, h, w, c] = np.asarray([0, 0, h, w, c], dtype=np.int32)
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1, 2, 3, 4, 5],
        shape_signature=[1, 1, 2, 3, 4, 5],
        data=indices,
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["updates"] = TensorIR(
        name="updates",
        dtype="FLOAT32",
        shape=[1, 4, 1, 2, 3],
        shape_signature=[1, 4, 1, 2, 3],
        logical_layout="NCDHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 1, 2, 3, 4], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 1, 2, 3],
        shape_signature=[1, 4, 1, 2, 3],
        logical_layout="NCDHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=["indices", "updates", "shape"],
            outputs=["y"],
            options={},
        )
    )
    return model_ir


def _make_gather_elements_axis_coord_unsqueeze_model_ir() -> ModelIR:
    model_ir = ModelIR(name="gather_elements_axis_coord_unsqueeze_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["wa/GatherElements_output_0_gather_elements_axis_coord"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["k"] = TensorIR(
        name="k",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["wa/TopK_output_0_topk_values_raw"] = TensorIR(
        name="wa/TopK_output_0_topk_values_raw",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["wa/TopK_output_1_topk_indices_raw"] = TensorIR(
        name="wa/TopK_output_1_topk_indices_raw",
        dtype="INT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["wa/GatherElements_output_0_gather_elements_indices_i32"] = TensorIR(
        name="wa/GatherElements_output_0_gather_elements_indices_i32",
        dtype="INT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["wa/GatherElements_output_0_gather_elements_axis_coord_reshape_shape"] = TensorIR(
        name="wa/GatherElements_output_0_gather_elements_axis_coord_reshape_shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 2, 3, 4, 1], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["wa/GatherElements_output_0_gather_elements_axis_coord"] = TensorIR(
        name="wa/GatherElements_output_0_gather_elements_axis_coord",
        dtype="INT32",
        shape=[1, 2, 3, 4, 1],
        shape_signature=[1, 2, 3, 4, 1],
        logical_layout="NCDHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="TOPK_V2",
                inputs=["x", "k"],
                outputs=["wa/TopK_output_0_topk_values_raw", "wa/TopK_output_1_topk_indices_raw"],
                options={"axis": -1, "largest": True, "sorted": True},
            ),
            OperatorIR(
                op_type="CAST",
                inputs=["wa/TopK_output_1_topk_indices_raw"],
                outputs=["wa/GatherElements_output_0_gather_elements_indices_i32"],
                options={"inDataType": "INT32", "outDataType": "INT32"},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=[
                    "wa/GatherElements_output_0_gather_elements_indices_i32",
                    "wa/GatherElements_output_0_gather_elements_axis_coord_reshape_shape",
                ],
                outputs=["wa/GatherElements_output_0_gather_elements_axis_coord"],
                options={"newShape": [1, 2, 3, 4, 1]},
            ),
        ]
    )
    return model_ir


def _make_gather_elements_dynamic_coords_model_ir() -> ModelIR:
    model_ir = ModelIR(name="gather_elements_dynamic_coords_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    grid = np.indices((1, 2, 3, 4), dtype=np.int32)
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["k"] = TensorIR(
        name="k",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["wa/TopK_output_0_topk_values_raw"] = TensorIR(
        name="wa/TopK_output_0_topk_values_raw",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["wa/TopK_output_1_topk_indices_raw"] = TensorIR(
        name="wa/TopK_output_1_topk_indices_raw",
        dtype="INT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["wa/GatherElements_output_0_gather_elements_indices_i32"] = TensorIR(
        name="wa/GatherElements_output_0_gather_elements_indices_i32",
        dtype="INT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["wa/GatherElements_output_0_gather_elements_axis_coord_reshape_shape"] = TensorIR(
        name="wa/GatherElements_output_0_gather_elements_axis_coord_reshape_shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 2, 3, 4, 1], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["wa/GatherElements_output_0_gather_elements_axis_coord"] = TensorIR(
        name="wa/GatherElements_output_0_gather_elements_axis_coord",
        dtype="INT32",
        shape=[1, 2, 3, 4, 1],
        shape_signature=[1, 2, 3, 4, 1],
        logical_layout="UNKNOWN",
    )
    for dim in range(3):
        model_ir.tensors[f"wa/GatherElements_output_0_gather_elements_coord_{dim}"] = TensorIR(
            name=f"wa/GatherElements_output_0_gather_elements_coord_{dim}",
            dtype="INT32",
            shape=[1, 2, 3, 4, 1],
            shape_signature=[1, 2, 3, 4, 1],
            data=np.expand_dims(grid[dim], axis=-1),
            logical_layout="UNKNOWN",
        )
    model_ir.tensors["wa/GatherElements_output_0_gather_elements_coords"] = TensorIR(
        name="wa/GatherElements_output_0_gather_elements_coords",
        dtype="INT32",
        shape=[1, 2, 3, 4, 4],
        shape_signature=[1, 2, 3, 4, 4],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="TOPK_V2",
                inputs=["x", "k"],
                outputs=["wa/TopK_output_0_topk_values_raw", "wa/TopK_output_1_topk_indices_raw"],
                options={"axis": -1, "largest": True, "sorted": True},
            ),
            OperatorIR(
                op_type="CAST",
                inputs=["wa/TopK_output_1_topk_indices_raw"],
                outputs=["wa/GatherElements_output_0_gather_elements_indices_i32"],
                options={"inDataType": "INT32", "outDataType": "INT32"},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=[
                    "wa/GatherElements_output_0_gather_elements_indices_i32",
                    "wa/GatherElements_output_0_gather_elements_axis_coord_reshape_shape",
                ],
                outputs=["wa/GatherElements_output_0_gather_elements_axis_coord"],
                options={"newShape": [1, 2, 3, 4, 1]},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[
                    "wa/GatherElements_output_0_gather_elements_coord_0",
                    "wa/GatherElements_output_0_gather_elements_coord_1",
                    "wa/GatherElements_output_0_gather_elements_coord_2",
                    "wa/GatherElements_output_0_gather_elements_axis_coord",
                ],
                outputs=["wa/GatherElements_output_0_gather_elements_coords"],
                options={"axis": 4, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="GATHER_ND",
                inputs=["x", "wa/GatherElements_output_0_gather_elements_coords"],
                outputs=["y"],
                options={},
            ),
        ]
    )
    return model_ir


def _make_dynamic_full_sort_topk_model_ir() -> ModelIR:
    model_ir = ModelIR(name="dynamic_full_sort_topk_model_ir")
    model_ir.inputs = ["scores", "positions"]
    model_ir.outputs = ["sorted_values", "sorted_indices", "sorted_positions"]
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["positions"] = TensorIR(
        name="positions",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["positions_shape"] = TensorIR(
        name="positions_shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["axis0"] = TensorIR(
        name="axis0",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["k_vec"] = TensorIR(
        name="k_vec",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["k"] = TensorIR(
        name="k",
        dtype="INT32",
        shape=[],
        shape_signature=[],
    )
    model_ir.tensors["sorted_values"] = TensorIR(
        name="sorted_values",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["sorted_indices"] = TensorIR(
        name="sorted_indices",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["sorted_positions"] = TensorIR(
        name="sorted_positions",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="SHAPE",
                inputs=["positions"],
                outputs=["positions_shape"],
                options={"outType": "INT32"},
            ),
            OperatorIR(
                op_type="GATHER",
                inputs=["positions_shape", "axis0"],
                outputs=["k_vec"],
                options={"axis": 0, "batchDims": 0},
            ),
            OperatorIR(
                op_type="SQUEEZE",
                inputs=["k_vec"],
                outputs=["k"],
                options={"squeezeDims": [0]},
            ),
            OperatorIR(
                op_type="TOPK_V2",
                inputs=["scores", "k"],
                outputs=["sorted_values", "sorted_indices"],
                options={"axis": -1, "largest": True, "sorted": True},
            ),
            OperatorIR(
                op_type="GATHER",
                inputs=["positions", "sorted_indices"],
                outputs=["sorted_positions"],
                options={"axis": 0, "batchDims": 0},
            ),
        ]
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


def _make_concat_unknown_peer_layout_model_ir() -> ModelIR:
    model_ir = ModelIR(name="concat_unknown_peer_layout_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 128],
        shape_signature=[1, 1, 128],
        logical_layout="NCW",
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[128],
        shape_signature=[128],
        data=np.zeros((128,), dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["seq"] = TensorIR(
        name="seq",
        dtype="FLOAT32",
        shape=[1, 1, 128],
        shape_signature=[1, 1, 128],
        logical_layout="NCW",
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 16, 8, 1], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["branch_unknown"] = TensorIR(
        name="branch_unknown",
        dtype="FLOAT32",
        shape=[1, 16, 8, 1],
        shape_signature=[1, 16, 8, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["branch_known"] = TensorIR(
        name="branch_known",
        dtype="FLOAT32",
        shape=[1, 16, 8, 1],
        shape_signature=[1, 16, 8, 1],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 32, 8, 1],
        shape_signature=[1, 32, 8, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="ADD",
                inputs=["x", "bias"],
                outputs=["seq"],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["seq", "reshape_shape"],
                outputs=["branch_unknown"],
                options={"newShape": [1, 16, 8, 1]},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["branch_unknown", "branch_known"],
                outputs=["y"],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            ),
        ]
    )
    return model_ir


def _make_nhwc_public_input_boundary_model_ir() -> ModelIR:
    model_ir = ModelIR(name="nhwc_public_input_boundary_model_ir")
    model_ir.inputs = ["image"]
    model_ir.outputs = ["features"]
    model_ir.tensors["image"] = TensorIR(
        name="image",
        dtype="FLOAT32",
        shape=[1, 96, 1408, 3],
        shape_signature=[-1, 96, 1408, 3],
    )
    model_ir.tensors["scale"] = TensorIR(
        name="scale",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(0.5, dtype=np.float32),
    )
    model_ir.tensors["image_scaled"] = TensorIR(
        name="image_scaled",
        dtype="FLOAT32",
        shape=[1, 96, 1408, 3],
        shape_signature=[-1, 96, 1408, 3],
    )
    model_ir.tensors["image_nchw"] = TensorIR(
        name="image_nchw",
        dtype="FLOAT32",
        shape=[1, 3, 96, 1408],
        shape_signature=[-1, 3, 96, 1408],
    )
    model_ir.tensors["features"] = TensorIR(
        name="features",
        dtype="FLOAT32",
        shape=[1, 3, 96, 1408],
        shape_signature=[-1, 3, 96, 1408],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="MUL",
                inputs=["image", "scale"],
                outputs=["image_scaled"],
                options={},
            ),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["image_scaled"],
                outputs=["image_nchw"],
                options={"perm": [0, 3, 1, 2]},
            ),
            OperatorIR(
                op_type="IDENTITY",
                inputs=["image_nchw"],
                outputs=["features"],
                options={},
            ),
        ]
    )
    return model_ir


def _make_nhwc_public_output_boundary_model_ir() -> ModelIR:
    model_ir = ModelIR(name="nhwc_public_output_boundary_model_ir")
    model_ir.inputs = ["features"]
    model_ir.outputs = ["image"]
    model_ir.tensors["features"] = TensorIR(
        name="features",
        dtype="FLOAT32",
        shape=[1, 3, 96, 1408],
        shape_signature=[-1, 3, 96, 1408],
    )
    model_ir.tensors["features_id"] = TensorIR(
        name="features_id",
        dtype="FLOAT32",
        shape=[1, 3, 96, 1408],
        shape_signature=[-1, 3, 96, 1408],
    )
    model_ir.tensors["image"] = TensorIR(
        name="image",
        dtype="FLOAT32",
        shape=[1, 96, 1408, 3],
        shape_signature=[-1, 96, 1408, 3],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="IDENTITY",
                inputs=["features"],
                outputs=["features_id"],
                options={},
            ),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["features_id"],
                outputs=["image"],
                options={"perm": [0, 2, 3, 1]},
            ),
        ]
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


def test_normalize_model_ir_shrinks_internal_channel_last_conv_relu_conv_island() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(_make_boundary_wrapped_conv_relu_conv_model_ir())
    assert normalize_logical_layout(normalized.tensors["x_nhwc"].logical_layout) == "NHWC"
    assert normalize_logical_layout(normalized.tensors["conv0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["relu0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["y_nhwc"].logical_layout) == "NHWC"
    assert sum(1 for op in normalized.operators if str(op.op_type) == "TRANSPOSE") == 2


def test_normalize_model_ir_shrinks_internal_channel_last_conv_pool_conv_island() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(_make_boundary_wrapped_conv_pool_conv_model_ir())
    assert normalize_logical_layout(normalized.tensors["x_nhwc"].logical_layout) == "NHWC"
    assert normalize_logical_layout(normalized.tensors["conv0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["pool0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["y_nhwc"].logical_layout) == "NHWC"
    assert sum(1 for op in normalized.operators if str(op.op_type) == "TRANSPOSE") == 2


def test_normalize_model_ir_shrinks_dual_pool_concat_conv_island() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(
        _make_boundary_wrapped_conv_dual_pool_concat_conv_model_ir()
    )
    assert normalize_logical_layout(normalized.tensors["x_nhwc"].logical_layout) == "NHWC"
    assert normalize_logical_layout(normalized.tensors["conv0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["max0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["avg0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["cat0_nhwc"].logical_layout) == "NCHW"
    assert normalize_logical_layout(normalized.tensors["y_nhwc"].logical_layout) == "NHWC"
    concat_op = next(op for op in normalized.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options["axis"]) == 1
    assert sum(1 for op in normalized.operators if str(op.op_type) == "TRANSPOSE") == 2


def test_normalize_model_ir_preserves_channel_last_gather_branch() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(_make_boundary_wrapped_gather_branch_model_ir())
    assert normalize_logical_layout(normalized.tensors["x_nhwc"].logical_layout) == "NHWC"
    assert normalize_logical_layout(normalized.tensors["gathered_nhwc"].logical_layout) == "NHWC"
    assert sum(1 for op in normalized.operators if str(op.op_type) == "TRANSPOSE") == 2


def test_export_pytorch_package_runs_residual_add_island_on_channel_first_aliases(tmp_path) -> None:
    model_ir = _make_boundary_wrapped_conv_add_conv_model_ir()
    normalized = prepare_model_ir_for_native_pytorch(model_ir)
    package_path = str(tmp_path / "boundary_wrapped_conv_add_conv_pytorch")
    metadata = _build_metadata_payload(normalized)
    metadata["execution_backend"] = "native"
    tensor_storage_name_map = _make_tensor_storage_name_map(normalized)
    _write_native_model_file(
        package_path,
        model_ir=normalized,
        metadata=metadata,
        tensor_storage_name_map=tensor_storage_name_map,
    )
    package_dir = Path(package_path)
    model_source = (package_dir / "model.py").read_text(encoding="utf-8")

    assert re.search(r"\w+_cf = torch\.add\(\w+_cf, \w+_cf\)", model_source) is not None
    assert re.search(r"self\.conv_block_2\(\w+_cf\)", model_source) is not None
    assert "torch.add(conv1_nhwc, conv0_nhwc)" not in model_source


def test_build_tensor_var_name_map_shortens_long_tensor_names_deterministically() -> None:
    long_name = (
        "model_BLOCK_4_4_BN_2_FusedBatchNormV3_model_BLOCK_4_1_CONV_1_Conv2D_"
        "model_BLOCK_4_4_CONV_2_depthwise1_input_nhwc__channel_first"
    )
    sibling_name = (
        "model_BLOCK_4_4_BN_2_FusedBatchNormV3_model_BLOCK_4_1_CONV_1_Conv2D_"
        "model_BLOCK_4_4_CONV_2_depthwise1_output_nhwc__channel_first"
    )
    model_ir = ModelIR(name="shorten_tensor_var_names")
    model_ir.inputs = ["x"]
    model_ir.outputs = [long_name, sibling_name]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
    )
    model_ir.tensors[long_name] = TensorIR(
        name=long_name,
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
    )
    model_ir.tensors[sibling_name] = TensorIR(
        name=sibling_name,
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
    )

    first_mapping = _build_tensor_var_name_map(model_ir)
    second_mapping = _build_tensor_var_name_map(model_ir)

    assert first_mapping == second_mapping
    assert len(first_mapping[long_name]) <= 40
    assert len(first_mapping[sibling_name]) <= 40
    assert re.search(r"_cf(?:_[0-9a-f]{4})?$", first_mapping[long_name]) is not None
    assert re.search(r"_cf(?:_[0-9a-f]{4})?$", first_mapping[sibling_name]) is not None
    assert "channel_first" not in first_mapping[long_name].lower()
    assert "channel_first" not in first_mapping[sibling_name].lower()
    assert "fusedbatchnormv3" not in first_mapping[long_name].lower()
    assert first_mapping[long_name] != first_mapping[sibling_name]


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

    assert normalized.tensors["y"].shape == [1, 8, 3]
    assert np.asarray(normalized.tensors["shape"].data).reshape(-1).tolist() == [1, 8, 3]
    reshape_op = next(op for op in normalized.operators if str(op.op_type) == "RESHAPE")
    assert reshape_op.options["newShape"] == [1, 8, 3]


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
    assert "class _Conv2dBlock(torch.nn.Module):" not in model_source
    assert "    _Conv2dBlock,\n" in model_source
    assert "self.conv_block_0 = _Conv2dBlock(" in model_source
    assert "LOAD_SPECS" not in model_source
    assert "_copy_tensor_data" not in model_source
    assert "_validate_state_dict_keys" not in model_source
    assert "class _Conv2dBlock(torch.nn.Module):" in runtime_source
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


def test_export_runtime_wrapper_package_supports_nhwc_pool2d(tmp_path) -> None:
    package_path = _export_runtime_wrapper_package_from_model_ir(
        model_ir=_make_runtime_wrapper_pool_nhwc_model_ir(),
        output_folder_path=str(tmp_path / "pool_nhwc_runtime_wrapper"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 4 * 6 * 2, dtype=torch.float32).reshape(1, 4, 6, 2)
    out = model(x)
    expected = F.max_pool2d(
        x.permute(0, 3, 1, 2),
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=0,
    ).permute(0, 2, 3, 1).contiguous()
    assert out.shape == torch.Size([1, 2, 3, 2])
    assert torch.allclose(out, expected)


def test_export_generated_package_keeps_same_shape_nchw_pool_in_exported_program(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_same_shape_max_pool_codegen_model_ir(),
        output_folder_path=str(tmp_path / "same_shape_pool_codegen"),
    )
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    permute_targets = [
        node
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.permute.default"
    ]
    assert permute_targets == []

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 256 * 15 * 20, dtype=torch.float32).reshape(1, 256, 15, 20)
    out = model(x)
    expected = F.max_pool2d(x, kernel_size=(5, 5), stride=(1, 1), padding=2)
    assert torch.allclose(out, expected)


def test_export_generated_package_folds_single_use_static_reshape_chain(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_nested_static_reshape_model_ir(),
        output_folder_path=str(tmp_path / "nested_static_reshape_codegen"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "mid = torch.reshape(" not in model_source

    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    reshape_nodes = [
        node
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.reshape.default"
    ]
    assert len(reshape_nodes) == 1

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5)
    out = model(x)
    expected = torch.reshape(x, [1, 60])
    assert torch.equal(out, expected)


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
    runtime_source = (Path(package_path) / "runtime.py").read_text(encoding="utf-8")
    assert "class _Conv2dBlock(torch.nn.Module):" not in model_source
    assert "    _Conv2dBlock,\n" in model_source
    assert "self.conv_block_0 = _Conv2dBlock(" in model_source
    assert "class _Conv2dBlock(torch.nn.Module):" in runtime_source
    assert "return torch.relu(x)" in runtime_source


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
    assert "class _Conv2dBlock(torch.nn.Module):" not in model_source
    assert "    _Conv2dBlock,\n" in model_source
    assert "self.conv_block_0 = _Conv2dBlock(" in model_source
    assert "self.conv_block_1 = _Conv2dBlock(" in model_source
    assert "_apply_module_conv2d" not in model_source
    assert "_forward_stage_" not in model_source
    assert "class _Conv2dBlock(torch.nn.Module):" in runtime_source
    assert "return torch.relu(x)" in runtime_source
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


def test_normalize_model_ir_preserves_channel_last_factorized_rank3_detection_head_sequence() -> None:
    model_ir = ModelIR(name="channel_last_factorized_rank3_detection_head")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["scores"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 6, 3, 4],
        shape_signature=[1, 6, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["x_cl"] = TensorIR(
        name="x_cl",
        dtype="FLOAT32",
        shape=[1, 3, 4, 6],
        shape_signature=[1, 3, 4, 6],
        logical_layout="NHWC",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 36, 2], dtype=np.int32),
    )
    model_ir.tensors["seq"] = TensorIR(
        name="seq",
        dtype="FLOAT32",
        shape=[1, 36, 2],
        shape_signature=[1, 36, 2],
        logical_layout="NWC",
    )
    model_ir.tensors["soft"] = TensorIR(
        name="soft",
        dtype="FLOAT32",
        shape=[1, 36, 2],
        shape_signature=[1, 36, 2],
        logical_layout="NWC",
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1, 2, 36],
        shape_signature=[1, 2, 36],
        logical_layout="NCW",
    )
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_cl"],
            options={"perm": [0, 2, 3, 1]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_cl", "shape"],
            outputs=["seq"],
            options={"newShape": [1, 36, 2], "onnxRawNewShape": [1, 36, 2]},
        ),
        OperatorIR(
            op_type="SOFTMAX",
            inputs=["seq"],
            outputs=["soft"],
            options={"axis": 2, "beta": 1.0},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["soft"],
            outputs=["scores"],
            options={"perm": [0, 2, 1]},
        ),
    ]

    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)

    assert normalized.tensors["seq"].shape == [1, 36, 2]
    reshape_op = next(op for op in normalized.operators if str(op.outputs[0]) == "seq")
    softmax_op = next(op for op in normalized.operators if str(op.outputs[0]) == "soft")
    assert reshape_op.inputs[0] == "x"
    assert reshape_op.options["newShape"] == [1, 36, 2]
    assert softmax_op.options["axis"] == 2


def test_export_pytorch_package_preserves_channel_last_factorized_rank3_detection_head_sequence(tmp_path) -> None:
    model_ir = ModelIR(name="channel_last_factorized_rank3_detection_head_export")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["scores"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 6, 3, 4],
        shape_signature=[1, 6, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["x_cl"] = TensorIR(
        name="x_cl",
        dtype="FLOAT32",
        shape=[1, 3, 4, 6],
        shape_signature=[1, 3, 4, 6],
        logical_layout="NHWC",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 36, 2], dtype=np.int32),
    )
    model_ir.tensors["seq"] = TensorIR(
        name="seq",
        dtype="FLOAT32",
        shape=[1, 36, 2],
        shape_signature=[1, 36, 2],
        logical_layout="NWC",
    )
    model_ir.tensors["soft"] = TensorIR(
        name="soft",
        dtype="FLOAT32",
        shape=[1, 36, 2],
        shape_signature=[1, 36, 2],
        logical_layout="NWC",
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1, 2, 36],
        shape_signature=[1, 2, 36],
        logical_layout="NCW",
    )
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_cl"],
            options={"perm": [0, 2, 3, 1]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_cl", "shape"],
            outputs=["seq"],
            options={"newShape": [1, 36, 2], "onnxRawNewShape": [1, 36, 2]},
        ),
        OperatorIR(
            op_type="SOFTMAX",
            inputs=["seq"],
            outputs=["soft"],
            options={"axis": 2, "beta": 1.0},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["soft"],
            outputs=["scores"],
            options={"perm": [0, 2, 1]},
        ),
    ]

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "factorized_rank3_detection_head_sequence"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "[1, 36, 2]" in model_source
    assert "axis=2" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(72, dtype=torch.float32).reshape(1, 6, 3, 4)
    out = model(x)
    ref = torch.softmax(x.permute(0, 2, 3, 1).contiguous().reshape(1, 36, 2), dim=2).permute(0, 2, 1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_reshape_preserves_runtime_minus_one_shape_tensor(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_preserves_runtime_minus_one_shape_tensor")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([-1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
    )
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={"newShape": []})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_runtime_minus_one_shape_tensor_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.arange(20, dtype=torch.float32)
    out = model(x)
    assert out.shape == (20, 1)
    assert torch.equal(out, x.reshape(20, 1))
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "[int(v) for v in [1, 1]]" not in model_source
    assert "_resolve_reshape_shape([-1, 1], x, allow_zero=False)" in model_source


def test_export_pytorch_package_reshape_detection_head_packed_channels_to_rank5(tmp_path) -> None:
    rng = np.random.default_rng(19)
    model_ir = ModelIR(name="reshape_detection_head_packed_channels_to_rank5")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 18, 5, 7],
        shape_signature=[1, 18, 5, 7],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 5, 7, 3, 6], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 7, 3, 6],
        shape_signature=[1, 5, 7, 3, 6],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_detection_head_rank5_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.from_numpy(rng.standard_normal((1, 18, 5, 7), dtype=np.float32))
    out = model(x)
    expected = x.permute(0, 2, 3, 1).contiguous().reshape(1, 5, 7, 3, 6)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)
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


def test_normalize_model_ir_allows_transpose_sandwiched_last_axis_softmax() -> None:
    model_ir = ModelIR(name="transpose_sandwiched_softmax")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 48, 80],
        shape_signature=[1, 2, 48, 80],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_to_last"] = TensorIR(
        name="perm_to_last",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 2, 1], dtype=np.int32),
    )
    model_ir.tensors["scores_logits"] = TensorIR(
        name="scores_logits",
        dtype="FLOAT32",
        shape=[1, 80, 48, 2],
        shape_signature=[1, 80, 48, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["scores_prob"] = TensorIR(
        name="scores_prob",
        dtype="FLOAT32",
        shape=[1, 80, 48, 2],
        shape_signature=[1, 80, 48, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["perm_from_last"] = TensorIR(
        name="perm_from_last",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 2, 1], dtype=np.int32),
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 2, 48, 80],
        shape_signature=[1, 2, 48, 80],
        logical_layout="NCHW",
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_to_last"], outputs=["scores_logits"], options={}),
        OperatorIR(op_type="SOFTMAX", inputs=["scores_logits"], outputs=["scores_prob"], options={"beta": 1.0}),
        OperatorIR(op_type="TRANSPOSE", inputs=["scores_prob", "perm_from_last"], outputs=["z"], options={}),
    ]

    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)
    assert normalized.tensors["scores_logits"].logical_layout == "UNKNOWN"
    assert normalized.tensors["scores_prob"].logical_layout == "UNKNOWN"
    assert normalized.tensors["z"].shape == [1, 2, 48, 80]


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
    assert metadata["execution_backend"] == "native"
    assert "load_generated_model_package" not in model_source
    assert "TENSOR_STORAGE_NAMES" not in model_source
    assert "resolve_model_tensor" not in model_source
    assert "_torch_dtype(" not in model_source
    assert "cast(torch.Tensor, self." not in model_source
    assert "def _init_constants(self) -> None:" in model_source
    assert model_source.count("_apply_") <= 40
    assert "_forward_stage_0" in model_source
    assert "cv76_in = _apply_concat([onnx_concat199, resize72_out], axis=3" not in model_source
    assert "cv18_in = torch.reshape(_apply_concat(" not in model_source
    assert "resize72_in = _align_tensor_to_target_shape(resize72_in_cf, [1, 15, 20, 128])" not in model_source
    assert "resize90_in = _align_tensor_to_target_shape(resize90_in_cf, [1, 30, 40, 64])" not in model_source

    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    cat_6 = next(
        (node for node in exported_program.module().graph.nodes if node.op == "call_function" and node.name == "cat_6"),
        None,
    )
    assert cat_6 is not None
    cat_6_inputs = list(cat_6.args[0]) if len(cat_6.args) >= 1 else []
    assert all(
        str(getattr(input_node, "target", "")) != "aten.permute.default"
        for input_node in cat_6_inputs
    )
    assert all(
        str(getattr(user_node, "target", "")) != "aten.permute.default"
        for user_node in cat_6.users
    )
    inverse_permute_pairs = set()
    nested_reshape_shape_pairs = set()
    for node in exported_program.module().graph.nodes:
        if node.op == "call_function" and str(node.target) == "aten.permute.default":
            source = node.args[0] if len(node.args) >= 1 else None
            if getattr(source, "op", None) != "call_function" or str(getattr(source, "target", "")) != "aten.contiguous.default":
                continue
            nested = source.args[0] if len(source.args) >= 1 else None
            if getattr(nested, "op", None) != "call_function" or str(getattr(nested, "target", "")) != "aten.permute.default":
                continue
            perm_a = list(nested.args[1]) if len(nested.args) >= 2 else None
            perm_b = list(node.args[1]) if len(node.args) >= 2 else None
            if perm_a is None or perm_b is None:
                continue
            if [perm_a[int(idx)] for idx in perm_b] == list(range(len(perm_a))):
                inverse_permute_pairs.add((nested.name, node.name))
        if node.op == "call_function" and str(node.target) == "aten.reshape.default":
            source = node.args[0] if len(node.args) >= 1 else None
            if getattr(source, "op", None) == "call_function" and str(getattr(source, "target", "")) == "aten.reshape.default":
                source_val = source.meta.get("val")
                target_val = node.meta.get("val")
                source_shape = str(tuple(source_val.shape)) if hasattr(source_val, "shape") else "None"
                target_shape = str(tuple(target_val.shape)) if hasattr(target_val, "shape") else "None"
                nested_reshape_shape_pairs.add((source_shape, target_shape))
    assert inverse_permute_pairs == set()
    assert nested_reshape_shape_pairs == {
        ("(u0,)", "(u0, 1)"),
    }


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
    assert metadata["execution_backend"] == "native"
    assert "load_generated_model_package" not in model_source
    assert "class _Conv2dBlock(torch.nn.Module):" not in model_source
    assert "_apply_gather(" not in model_source
    assert "_apply_gather_nd(" not in model_source
    assert "_apply_slice(" not in model_source
    assert "_apply_strided_slice(" not in model_source
    assert "t798_cf = torch.cat([cv261_out, t_796, t_797], dim=1)" in model_source
    assert "t824_cf = torch.cat([cv282_out, t_822, t_823], dim=1)" in model_source
    assert "t850_cf = torch.cat([cv303_out, t_848, t_849], dim=1)" in model_source
    assert "t_798 = _align_tensor_to_target_shape(t798_cf.permute(0, 2, 3, 1).contiguous()" not in model_source
    assert "t_824 = _align_tensor_to_target_shape(t824_cf.permute(0, 2, 3, 1).contiguous()" not in model_source
    assert "t_850 = _align_tensor_to_target_shape(t850_cf.permute(0, 2, 3, 1).contiguous()" not in model_source
    assert model_source.count("register_buffer(") <= 12
    assert "    _Conv2dBlock,\n" in model_source
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

    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    inverse_permute_pairs = set()
    for node in exported_program.module().graph.nodes:
        if node.op != "call_function" or str(node.target) != "aten.permute.default":
            continue
        source = node.args[0] if len(node.args) >= 1 else None
        if getattr(source, "op", None) != "call_function" or str(getattr(source, "target", "")) != "aten.contiguous.default":
            continue
        nested = source.args[0] if len(source.args) >= 1 else None
        if getattr(nested, "op", None) != "call_function" or str(getattr(nested, "target", "")) != "aten.permute.default":
            continue
        perm_a = list(nested.args[1]) if len(nested.args) >= 2 else None
        perm_b = list(node.args[1]) if len(node.args) >= 2 else None
        if perm_a is None or perm_b is None:
            continue
        if [perm_a[int(idx)] for idx in perm_b] == list(range(len(perm_a))):
            inverse_permute_pairs.add((nested.name, node.name))
    assert inverse_permute_pairs == set()


def test_export_pytorch_package_folds_public_input_focus_bridge_for_yolox_when_model_is_available(tmp_path) -> None:
    model_path = Path("yolox_s.onnx")
    if not model_path.exists():
        pytest.skip("yolox_s.onnx is not available")
    model_proto = onnx.load(model_path)
    model_ir = clone_model_ir_with_float32(
        lower_onnx_to_ir(
            model_proto,
            output_file_name="yolox_public_bridge_codegen_test",
            show_progress=False,
            transpose_inputs_to_nhwc=True,
        )
    )
    prune_identity_cast_operators(model_ir, preserve_model_outputs=True)
    optimize_redundant_transpose_operators(model_ir, preserve_model_outputs=True)
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "yolox_public_bridge_pytorch"),
    )
    package_dir = Path(package_path)
    model_source = (package_dir / "model.py").read_text()
    assert "images_public_layout_bridge = _torch_permute(images, [0, 2, 3, 1])" not in model_source
    assert "t490_cf = images[0:1, 0:3, 0:640:2, 0:640:2]" in model_source
    assert "cv41_in = torch.cat([t490_cf, t501_cf, t496_cf, t512_cf], dim=1)" in model_source

    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    graph_nodes = list(exported_program.module().graph.nodes)
    image_placeholder_index = next(
        idx for idx, node in enumerate(graph_nodes)
        if node.op == "placeholder" and node.name == "images"
    )
    first_call_function = next(
        node for node in graph_nodes[image_placeholder_index + 1 :]
        if node.op == "call_function"
    )
    assert str(first_call_function.target) == "aten.slice.Tensor"


def test_export_pytorch_package_avoids_early_permute_chain_for_human_segmentation_when_model_is_available(tmp_path) -> None:
    model_path = Path("human_segmentation_pphumanseg_2021oct.onnx")
    if not model_path.exists():
        pytest.skip("human_segmentation_pphumanseg_2021oct.onnx is not available")
    model_proto = onnx.load(model_path)
    model_ir = clone_model_ir_with_float32(
        lower_onnx_to_ir(
            model_proto,
            output_file_name="human_segmentation_native_codegen_test",
            show_progress=False,
        )
    )
    prune_identity_cast_operators(model_ir, preserve_model_outputs=True)
    optimize_redundant_transpose_operators(model_ir, preserve_model_outputs=True)
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "human_segmentation_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert metadata["execution_backend"] == "native"
    assert "cv17_in_nhwc_cf.permute(0, 2, 3, 1).contiguous()" not in model_source
    assert "cv19_in_nhwc.permute(0, 3, 1, 2).contiguous()" not in model_source
    assert "cv20_out_nhwc_cf.permute(0, 2, 3, 1).contiguous()" not in model_source
    assert "cv21_in_nhwc.permute(0, 3, 1, 2).contiguous()" not in model_source
    assert "cv22_out_nhwc_cf.permute(0, 2, 3, 1).contiguous()" not in model_source

    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    permute_indices = [
        idx
        for idx, node in enumerate(exported_program.module().graph.nodes)
        if node.op == "call_function" and str(node.target) == "aten.permute.default"
    ]
    assert permute_indices
    assert min(permute_indices) > 320


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
    assert metadata["execution_backend"] == "native"
    assert "load_generated_model_package" not in model_source
    assert "F.gelu(" in model_source


def test_export_pytorch_package_imported_tflite_with_cumsum_stays_native(tmp_path) -> None:
    tflite_path = _write_model_ir_as_tflite(
        str(tmp_path),
        "cumsum_native",
        _make_cumsum_model_ir(),
    )
    imported_model_ir = import_model_ir_from_tflite(tflite_file_path=tflite_path)
    package_path = export_pytorch_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=str(tmp_path / "cumsum_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    assert metadata["execution_backend"] == "native"

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, -1.0, 5.0]], dtype=torch.float32)
    out = model(x)
    assert torch.allclose(out, torch.cumsum(x, dim=1))

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_bidirectional_sequence_lstm_stays_native(tmp_path) -> None:
    model_ir = _make_bidirectional_sequence_lstm_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "bilstm_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert metadata["execution_backend"] == "native"
    assert "_SequenceLSTMBlock" in model_source
    assert "torch.nn.LSTM(" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.randn(3, 1, 2, dtype=torch.float32)
    out = model(x)
    assert tuple(out.shape) == (3, 1, 4)
    assert torch.isfinite(out).all()

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_bidirectional_sequence_lstm_dynamic_batch_supports_dynamo_and_exported_program(
    tmp_path,
) -> None:
    model_ir = _make_bidirectional_sequence_lstm_model_ir()
    model_ir.tensors["x"].shape_signature = [3, -1, 2]
    model_ir.tensors["fw_h0"].shape_signature = [-1, 2]
    model_ir.tensors["fw_c0"].shape_signature = [-1, 2]
    model_ir.tensors["bw_h0"].shape_signature = [-1, 2]
    model_ir.tensors["bw_c0"].shape_signature = [-1, 2]
    model_ir.tensors["y"].shape_signature = [3, -1, 4]
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "bilstm_dynamic_native_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "sequence_length=3" in model_source
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        shape_hints=["x:3,1,2"],
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        shape_hints=["x:3,1,2"],
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_compact_bidirectional_sequence_lstm_stays_native(tmp_path) -> None:
    model_ir = _make_compact_bidirectional_sequence_lstm_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "compact_bidirectional_sequence_lstm_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert metadata["execution_backend"] == "native"
    assert "_SequenceLSTMBlock" in model_source
    assert "torch.nn.LSTM(" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.randn(3, 1, 2, dtype=torch.float32)
    out = model(x)
    assert tuple(out.shape) == (3, 1, 4)
    assert torch.isfinite(out).all()

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_reverse_lstm_from_onnx_stays_native(tmp_path) -> None:
    model_ir = lower_onnx_to_ir(
        _make_reverse_lstm_model(),
        output_file_name="reverse_lstm_native_codegen_test",
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reverse_lstm_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert metadata["execution_backend"] == "native"
    assert "_SequenceLSTMBlock" in model_source
    assert "torch.nn.LSTM(" in model_source
    assert "torch.flip(" in model_source
    assert ".reshape(-1).tolist()" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.randn(3, 1, 2, dtype=torch.float32)
    out = model(x)
    assert tuple(out.shape) == (3, 1, 1, 2)
    assert torch.isfinite(out).all()

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_bidirectional_sequence_rnn_stays_native(tmp_path) -> None:
    model_ir = _make_bidirectional_sequence_rnn_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "bidirectional_sequence_rnn_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert metadata["execution_backend"] == "native"
    assert "_SequenceRNNBlock" in model_source
    assert "torch.nn.RNN(" in model_source
    assert "torch.flip(" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.randn(3, 1, 2, dtype=torch.float32)
    out = model(x)
    assert tuple(out.shape) == (3, 1, 6)
    assert torch.isfinite(out).all()

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert Path(torchscript_path).exists()
    assert Path(dynamo_onnx_path).exists()
    assert Path(exported_program_path).exists()


def test_export_torchscript_handles_conv_transpose2d_module_helper(tmp_path) -> None:
    model_ir = lower_onnx_to_ir(
        _make_conv_transpose2d_model(),
        output_file_name="conv_transpose2d_torchscript_helper",
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv_transpose2d_torchscript_helper_pkg"),
    )
    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_random_standard_normal_stays_native(tmp_path) -> None:
    model_ir = ModelIR(name="random_standard_normal_native")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 42],
        shape_signature=[1, 1, 42],
        logical_layout="NCW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 42],
        shape_signature=[1, 1, 42],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SHAPE",
            inputs=["x"],
            outputs=["shape"],
            options={"outType": "INT32"},
        )
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RANDOM_STANDARD_NORMAL",
            inputs=["shape"],
            outputs=["y"],
            options={},
        )
    )
    infer_model_ir_logical_layouts(model_ir)

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "random_standard_normal_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    assert metadata["execution_backend"] == "native"

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.zeros((1, 1, 42), dtype=torch.float32)
    y = model(x)
    assert list(y.shape) == [1, 1, 42]
    assert y.dtype == torch.float32
    assert bool(torch.isfinite(y).all())

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_concat_torchscript_regression(tmp_path) -> None:
    model_ir = ModelIR(name="concat_torchscript_regression")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
        logical_layout="UNKNOWN",
    )
    model_ir.operators = [
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x", "y"],
            outputs=["z"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "concat_torchscript_regression"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    y = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    out = model(x, y)
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32))

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_elides_inconsistent_same_layout_transpose(tmp_path) -> None:
    model_ir = ModelIR(name="stale_same_layout_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 1],
        shape_signature=[1, 4, 1],
        logical_layout="NCW",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([2, 0, 1], dtype=np.int32),
    )
    model_ir.tensors["t"] = TensorIR(
        name="t",
        dtype="FLOAT32",
        shape=[1, 4, 1],
        shape_signature=[1, 4, 1],
        logical_layout="NCW",
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 1],
        shape_signature=[1, 4, 1],
        logical_layout="NCW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["t"], options={}),
            OperatorIR(op_type="ADD", inputs=["t", "bias"], outputs=["y"], options={}),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "stale_same_layout_transpose_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32)
    out = model(x)
    assert list(out.shape) == [1, 4, 1]
    assert torch.allclose(out, x + 1.0)


def test_export_pytorch_package_preserves_inconsistent_same_layout_transpose_before_reshape(tmp_path) -> None:
    model_ir = ModelIR(name="stale_same_layout_transpose_before_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["t"] = TensorIR(
        name="t",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 24], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 24],
        shape_signature=[1, 24],
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["t"], options={}),
            OperatorIR(op_type="RESHAPE", inputs=["t", "shape"], outputs=["y"], options={}),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "stale_same_layout_transpose_before_reshape_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert metadata["execution_backend"] == "native"
    assert ".permute(0, 2, 3, 1).contiguous()" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    out = model(x)
    expected = x.permute(0, 2, 3, 1).reshape(1, 24)
    assert list(out.shape) == [1, 24]
    assert torch.allclose(out, expected)


def test_export_pytorch_package_prefers_feature_last_reshape_for_adjx_batch_matmul(tmp_path) -> None:
    model_ir = ModelIR(name="adjx_batch_matmul_rank3")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 2],
        shape_signature=[1, 4, 2],
        logical_layout="NCW",
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([2, 1, 4], dtype=np.int32),
    )
    model_ir.tensors["r"] = TensorIR(
        name="r",
        dtype="FLOAT32",
        shape=[2, 1, 4],
        shape_signature=[2, 1, 4],
        logical_layout="NCW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[4, 6],
        shape_signature=[4, 6],
        data=(np.arange(24, dtype=np.float32).reshape(4, 6) / 10.0),
    )
    model_ir.tensors["m"] = TensorIR(
        name="m",
        dtype="FLOAT32",
        shape=[2, 6, 1],
        shape_signature=[2, 6, 1],
        logical_layout="NCW",
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[6],
        shape_signature=[6],
        data=np.asarray([0.5, -0.5, 0.25, -0.25, 1.0, -1.0], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[2, 6, 1],
        shape_signature=[2, 6, 1],
        logical_layout="NCW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="RESHAPE", inputs=["x", "reshape_shape"], outputs=["r"], options={"newShape": [2, 1, 4]}),
            OperatorIR(op_type="BATCH_MATMUL", inputs=["r", "w"], outputs=["m"], options={"adjX": True, "adjY": False}),
            OperatorIR(op_type="ADD", inputs=["m", "bias"], outputs=["y"], options={}),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "adjx_batch_matmul_rank3_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    out = model(x)
    ref_x = torch.reshape(x, (2, 4, 1)).transpose(-1, -2)
    ref = torch.matmul(ref_x, torch.as_tensor(model_ir.tensors["w"].data)) + torch.as_tensor(model_ir.tensors["bias"].data).reshape(1, 1, 6)
    ref = ref.permute(0, 2, 1).contiguous()
    assert list(out.shape) == [2, 6, 1]


def test_export_pytorch_package_preserves_feature_last_residual_norm_sequence(tmp_path) -> None:
    model_ir = _make_feature_last_residual_norm_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "feature_last_residual_norm_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert metadata["execution_backend"] == "native"
    assert "[int(v) for v in [3, 1, 8]]" in model_source
    assert "_torch_permute(seq, [1, 0, 2])" in model_source
    assert "[1, 3, 8]" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = model(x)

    ref = x.permute(1, 0, 2).contiguous().reshape(3, 8)
    ref = F.linear(
        ref,
        torch.as_tensor(model_ir.tensors["fc_w"].data).T.contiguous(),
        torch.as_tensor(model_ir.tensors["fc_b"].data),
    )
    ref = ref.reshape(3, 1, 8).permute(1, 0, 2).contiguous()
    ref = ref + torch.as_tensor(model_ir.tensors["bias"].data)
    ref = ref.mean(dim=2, keepdim=True)
    assert list(out.shape) == [1, 3, 1]
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


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


def test_export_pytorch_package_conv2d_permuted_hwc_input(tmp_path) -> None:
    model_ir = _make_conv2d_permuted_hwc_input_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_permuted_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".permute(0, 2, 3, 1).contiguous()" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 86 * 2 * 51), dtype=torch.float32).reshape(1, 86, 2, 51)
    out = model(x)
    w = torch.as_tensor(model_ir.tensors["w"].data).permute(0, 3, 1, 2)
    b = torch.as_tensor(model_ir.tensors["b"].data)
    ref = torch.nn.functional.conv2d(x.permute(0, 2, 3, 1).contiguous(), w, b, stride=1, padding=0)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_nhwc_same_conv_concat_supports_dynamo_and_exported_program(tmp_path) -> None:
    model_ir = _make_nhwc_same_conv_concat_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "nhwc_same_conv_concat_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "pad=[59, 60, 0, 0]" not in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 1 * 40 * 120), dtype=torch.float32).reshape(1, 1, 40, 120)
    out = model(x)
    w = torch.as_tensor(model_ir.tensors["w"].data).permute(0, 3, 1, 2)
    b = torch.as_tensor(model_ir.tensors["b"].data)
    conv_ref = torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2).contiguous(),
        w,
        b,
        stride=1,
        padding=0,
    ).permute(0, 2, 3, 1).contiguous()
    skip = torch.as_tensor(model_ir.tensors["skip"].data)
    ref = torch.cat([skip, conv_ref], dim=3)
    assert list(out.shape) == [1, 1, 40, 1024]
    assert float((out - ref).abs().max().item()) < 0.1
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_nhwc_same_depthwise_conv_preserves_spatial_padding_axes(tmp_path) -> None:
    model_ir = _make_nhwc_same_depthwise_conv_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "nhwc_same_depthwise_conv_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "pad=[2, 2, 0, 0]" not in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 6 * 160 * 256), dtype=torch.float32).reshape(1, 6, 160, 256)
    out = model(x)
    w = torch.as_tensor(model_ir.tensors["w"].data).permute(3, 0, 1, 2)
    b = torch.as_tensor(model_ir.tensors["b"].data)
    ref = torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2).contiguous(),
        w,
        b,
        stride=1,
        padding=2,
        groups=256,
    ).permute(0, 2, 3, 1).contiguous()
    assert list(out.shape) == [1, 6, 160, 256]
    assert float((out - ref).abs().max().item()) < 0.1


def test_export_pytorch_package_mirror_pad_trims_noop_outer_dims(tmp_path) -> None:
    model_ir = _make_mirror_pad_with_noop_outer_dims_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "mirror_pad_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 2 * 4 * 4), dtype=torch.float32).reshape(1, 2, 4, 4)
    out = model(x)
    ref = F.pad(x, [1, 1, 1, 1], mode="reflect")
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_mirror_pad_handles_non_suffix_axes(tmp_path) -> None:
    model_ir = _make_mirror_pad_with_non_suffix_axes_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "mirror_pad_non_suffix_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 4 * 2 * 5), dtype=torch.float32).reshape(1, 4, 2, 5)
    out = model(x)
    ref = F.pad(x.permute(0, 2, 1, 3).contiguous(), [2, 2, 1, 1], mode="reflect").permute(0, 2, 1, 3).contiguous()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_mirror_pad_constant_codegen_supports_dynamo_export(tmp_path) -> None:
    model_ir = _make_mirror_pad_with_non_suffix_axes_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "mirror_pad_non_suffix_export_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_apply_pad_nd(" not in model_source
    assert "mode='reflect'" in model_source

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)

    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


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


def test_export_pytorch_package_broadcasts_ncw_channelwise_constant_to_nchw(tmp_path) -> None:
    model_ir = ModelIR(name="channel_affine_rank_mismatch")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["gamma"] = TensorIR(
        name="gamma",
        dtype="FLOAT32",
        shape=[1, 3, 1],
        shape_signature=[1, 3, 1],
        logical_layout="NCW",
        data=np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="MUL", inputs=["gamma", "x"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "channel_affine_rank_mismatch_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".reshape([1, 3, 1, 1])" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 3 * 4 * 4), dtype=torch.float32).reshape(1, 3, 4, 4)
    out = model(x)
    ref = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).reshape(1, 3, 1, 1) * x
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


def test_export_pytorch_package_reshape_flattens_channel_first_tensor_to_rank2_without_reordering(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_channel_first_to_rank2_flatten")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[2, 3, 2, 2],
        shape_signature=[2, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 12], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[2, 12],
        shape_signature=[2, 12],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={},
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_channel_first_to_rank2_flatten"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".permute(0, 2, 3, 1).contiguous()" not in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    out = model(x)
    ref = x.reshape(2, 12)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_reshape_drops_singleton_axis_without_layout_reordering(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_drop_singleton_axis_without_reorder")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[5, 8, 8, 1],
        shape_signature=[5, 8, 8, 1],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([5, 8, 8], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[5, 8, 8],
        shape_signature=[5, 8, 8],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={},
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_drop_singleton_axis_without_reorder"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".permute(0, 2, 3, 1).contiguous()" not in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(5 * 8 * 8, dtype=torch.float32).reshape(5, 8, 8, 1)
    out = model(x)
    ref = x.reshape(5, 8, 8)
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
    ref = model.conv_block_0(x.permute(0, 1, 3, 2).contiguous())
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_elides_inconsistent_layout_transpose_before_conv(tmp_path) -> None:
    model_ir = ModelIR(name="conv1d_layout_bridge_cleanup")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 1, 8],
        shape_signature=[1, 4, 1, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 1, 8],
        shape_signature=[1, 4, 1, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[6, 1, 1, 4],
        shape_signature=[6, 1, 1, 4],
        data=np.arange(24, dtype=np.float32).reshape(6, 1, 1, 4),
        logical_layout="NHWC",
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[6],
        shape_signature=[6],
        data=np.zeros((6,), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 6, 1, 8],
        shape_signature=[1, 6, 1, 8],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["x", "perm"],
                outputs=["x_nhwc"],
                options={},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_nhwc", "w", "b"],
                outputs=["y"],
                options={
                    "strideH": 1,
                    "strideW": 1,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "SAME",
                    "fusedActivationFunction": "NONE",
                },
            ),
        ]
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv1d_layout_bridge_cleanup"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert "x_nhwc = x.permute" not in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 1, 8)
    out = model(x)
    ref = model.conv_block_0(x)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_infers_conv3d_ctor_from_imported_filter_layout(tmp_path) -> None:
    model_ir = ModelIR(name="conv3d_imported_filter_layout")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 64, 12, 12, 20],
        shape_signature=[1, 64, 12, 12, 20],
        logical_layout="NCDHW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 32, 1, 1, 64],
        shape_signature=[1, 32, 1, 1, 64],
        data=np.arange(1 * 32 * 1 * 1 * 64, dtype=np.float32).reshape(1, 32, 1, 1, 64),
        logical_layout="NDHWC",
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[32],
        shape_signature=[32],
        data=np.zeros((32,), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 32, 12, 12, 20],
        shape_signature=[1, 32, 12, 12, 20],
        logical_layout="NCDHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_3D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "strideD": 1,
                "strideH": 1,
                "strideW": 1,
                "dilationDFactor": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "padding": "SAME",
                "fusedActivationFunction": "NONE",
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv3d_imported_filter_layout"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "in_channels=64" in model_source
    assert "out_channels=32" in model_source
    assert "kernel_size=(1, 1, 1)" in model_source
    assert "groups=1" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 64 * 12 * 12 * 20, dtype=torch.float32).reshape(1, 64, 12, 12, 20)
    out = model(x)
    assert list(out.shape) == [1, 32, 12, 12, 20]

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    assert Path(torchscript_path).exists()
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["torchscript"]["trace_mode"] == "trace"


def test_export_dynamo_onnx_from_generated_package_writes_artifact_and_metadata(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_model_ir(),
        output_folder_path=str(tmp_path / "add_dynamo_pkg"),
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    exported_model = onnx.load(str(dynamo_onnx_path))

    def _assert_graph_and_node_metadata_cleared(graph: onnx.GraphProto) -> None:
        assert len(graph.metadata_props) == 0
        for node in graph.node:
            assert len(node.metadata_props) == 0
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _assert_graph_and_node_metadata_cleared(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        _assert_graph_and_node_metadata_cleared(subgraph)

    assert len(exported_model.metadata_props) == 0
    _assert_graph_and_node_metadata_cleared(exported_model.graph)
    assert len(exported_model.graph.input) == 2
    assert len(exported_model.graph.output) == 1
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dynamo_onnx"]["file_name"] == Path(dynamo_onnx_path).name
    assert metadata["dynamo_onnx"]["dynamic_inputs_present"] is False


def test_export_exported_program_from_generated_package_writes_artifact_and_metadata(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_model_ir(),
        output_folder_path=str(tmp_path / "add_exported_program_pkg"),
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()
    reloaded = torch.export.load(str(exported_program_path))
    assert reloaded is not None
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["exported_program"]["file_name"] == Path(exported_program_path).name
    assert metadata["exported_program"]["dynamic_inputs_present"] is False


def test_export_pytorch_package_preserves_conv3d_weight_layout_and_numeric_parity(tmp_path) -> None:
    model_ir = ModelIR(name="conv3d_numeric_parity")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    x = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 1, 3, 4, 5) / 10.0
    w = np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float32).reshape(2, 1, 2, 2, 2) / 5.0
    b = np.asarray([0.25, -0.5], dtype=np.float32)
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 3, 4, 5],
        shape_signature=[1, 1, 3, 4, 5],
        logical_layout="NCDHW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[2, 1, 2, 2, 2],
        shape_signature=[2, 1, 2, 2, 2],
        data=w,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[2],
        shape_signature=[2],
        data=b,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 2, 3, 4],
        shape_signature=[1, 2, 2, 3, 4],
        logical_layout="NCDHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_3D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "strideD": 1,
                "strideH": 1,
                "strideW": 1,
                "dilationDFactor": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "padding": "VALID",
                "fusedActivationFunction": "NONE",
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv3d_numeric_parity"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x_tensor = torch.as_tensor(x)
    expected = F.conv3d(x_tensor, torch.as_tensor(w), torch.as_tensor(b), stride=1, padding=0)
    actual = model(x_tensor)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    saved_state_dict = torch.load(Path(package_path) / "state_dict.pth", map_location="cpu")
    assert torch.equal(saved_state_dict["conv3d_0.weight"], torch.as_tensor(w))
    assert torch.equal(saved_state_dict["conv3d_0.bias"], torch.as_tensor(b))


def test_export_pytorch_package_preserves_conv3d_transpose_weight_layout_and_numeric_parity(tmp_path) -> None:
    model_ir = ModelIR(name="conv3d_transpose_numeric_parity")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    x = np.arange(1 * 1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 1, 2, 2, 2) / 10.0
    w = np.arange(1 * 2 * 2 * 2 * 2, dtype=np.float32).reshape(1, 2, 2, 2, 2) / 7.0
    b = np.asarray([0.5, -0.25], dtype=np.float32)
    model_ir.tensors["output_shape"] = TensorIR(
        name="output_shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 2, 3, 3, 3], dtype=np.int32),
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 2, 2, 2, 2],
        shape_signature=[1, 2, 2, 2, 2],
        data=w,
    )
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 2, 2, 2],
        shape_signature=[1, 1, 2, 2, 2],
        logical_layout="NCDHW",
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[2],
        shape_signature=[2],
        data=b,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 3, 3],
        shape_signature=[1, 2, 3, 3, 3],
        logical_layout="NCDHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_3D_TRANSPOSE",
            inputs=["output_shape", "w", "x", "b"],
            outputs=["y"],
            options={
                "strideD": 1,
                "strideH": 1,
                "strideW": 1,
                "padding": "VALID",
                "fusedActivationFunction": "NONE",
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv3d_transpose_numeric_parity"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x_tensor = torch.as_tensor(x)
    expected = F.conv_transpose3d(x_tensor, torch.as_tensor(w), torch.as_tensor(b), stride=1, padding=0)
    actual = model(x_tensor)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    saved_state_dict = torch.load(Path(package_path) / "state_dict.pth", map_location="cpu")
    assert torch.equal(saved_state_dict["conv_transpose3d_0.weight"], torch.as_tensor(w))
    assert torch.equal(saved_state_dict["conv_transpose3d_0.bias"], torch.as_tensor(b))


def test_export_pytorch_package_matches_lowered_onnx_conv3d_numeric_parity(tmp_path) -> None:
    onnx_model = _make_conv3d_model()
    model_ir = lower_onnx_to_ir(onnx_model, output_file_name="conv3d_numeric_parity")
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv3d_lowered_numeric_parity"),
    )

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 1, 3, 4, 5) / 10.0
    actual = model(torch.as_tensor(x))

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    expected = torch.as_tensor(session.run(None, {"x": x})[0])
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_matches_lowered_onnx_conv3d_transpose_numeric_parity(tmp_path) -> None:
    onnx_model = _make_conv_transpose3d_model()
    model_ir = lower_onnx_to_ir(onnx_model, output_file_name="conv3d_transpose_numeric_parity")
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv3d_transpose_lowered_numeric_parity"),
    )

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = np.arange(1 * 1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 1, 2, 2, 2) / 10.0
    actual = model(torch.as_tensor(x))

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    expected = torch.as_tensor(session.run(None, {"x": x})[0])
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_reshape_special_plan_nchw_to_ncdhw_preserves_values(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_nchw_to_ncdhw_special_plan")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 4, 3],
        shape_signature=[1, 1, 4, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 3, 1, 1, 4], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 1, 1, 4],
        shape_signature=[1, 3, 1, 1, 4],
        logical_layout="NCDHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={"newShape": [1, 3, 1, 1, 4]},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_nchw_to_ncdhw_special_plan"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 4, 3)
    actual = model(x)
    expected = x.permute(0, 3, 1, 2).reshape(1, 3, 1, 1, 4)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


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


def test_convert_flatbuffer_direct_outputs_dynamo_onnx_and_autogenerates_package(tmp_path) -> None:
    model_path = tmp_path / "add_dynamo.onnx"
    onnx.save(_make_add_model(), str(model_path))
    output_dir = tmp_path / "out_dynamo"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_dynamo_onnx=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "add_dynamo_pytorch"
    artifact_path = package_path / "add_dynamo_dynamo.onnx"
    assert (package_path / "state_dict.pth").exists()
    assert artifact_path.exists()
    metadata = json.loads((package_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dynamo_onnx"]["file_name"] == artifact_path.name


def test_convert_flatbuffer_direct_outputs_exported_program_and_autogenerates_package(tmp_path) -> None:
    model_path = tmp_path / "add_exported_program.onnx"
    onnx.save(_make_add_model(), str(model_path))
    output_dir = tmp_path / "out_exported_program"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_exported_program=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "add_exported_program_pytorch"
    artifact_path = package_path / "add_exported_program_ep.pt2"
    assert (package_path / "state_dict.pth").exists()
    assert artifact_path.exists()
    metadata = json.loads((package_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["exported_program"]["file_name"] == artifact_path.name


def test_export_generated_package_reduce_mean_constant_axes_outputs_dynamo_onnx_and_exported_program(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_reduce_mean_constant_axes_model_ir(),
        output_folder_path=str(tmp_path / "reduce_mean_constant_axes_pkg"),
    )

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )

    assert dynamo_onnx_path is not None
    assert exported_program_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert Path(exported_program_path).exists()
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dynamo_onnx"]["file_name"] == Path(dynamo_onnx_path).name
    assert metadata["exported_program"]["file_name"] == Path(exported_program_path).name


def test_convert_input_tflite_outputs_dynamo_onnx_and_exported_program(tmp_path) -> None:
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "add_input_exports", _make_add_model_ir())
    output_dir = tmp_path / "out_tflite_exports"
    onnx2tf.convert(
        input_tflite_file_path=str(tflite_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_dynamo_onnx=True,
        flatbuffer_direct_output_exported_program=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "add_input_exports_pytorch"
    assert (package_path / "add_input_exports_dynamo.onnx").exists()
    assert (package_path / "add_input_exports_ep.pt2").exists()


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


def test_export_tflite_model_flatbuffer_direct_split_pytorch_aux_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "split_aux_out"
    outputs = export_tflite_model_flatbuffer_direct(
        onnx_graph=_make_add_model(),
        output_folder_path=str(output_dir),
        output_file_name="split_add_aux",
        force_split_manifest=True,
        output_pytorch_from_model_ir=True,
        output_dynamo_onnx_from_model_ir=True,
        output_exported_program_from_model_ir=True,
        pytorch_output_folder_path=str(output_dir / "split_add_aux_pytorch"),
    )
    assert "split_pytorch_dynamo_onnx_paths" in outputs
    assert "split_pytorch_exported_program_paths" in outputs
    manifest = json.loads(Path(outputs["split_manifest_path"]).read_text(encoding="utf-8"))
    assert len(manifest["partitions"]) >= 1
    first_partition = manifest["partitions"][0]
    assert "pytorch_dynamo_onnx_file_name" in first_partition
    assert "pytorch_exported_program_file_name" in first_partition
    first_package = output_dir / first_partition["pytorch_package_dir"]
    assert (first_package / first_partition["pytorch_dynamo_onnx_file_name"]).exists()
    assert (first_package / first_partition["pytorch_exported_program_file_name"]).exists()


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


def test_export_pytorch_package_normalizes_stale_residual_layout_transpose(tmp_path) -> None:
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
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "residual_pytorch"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tensors"]["y"]["shape"] == [1, 3, 4, 4]
    assert metadata["tensors"]["y"]["logical_layout"] == "NCHW"
    assert [str(op["op_type"]) for op in metadata["operators"]] == ["IDENTITY", "IDENTITY"]


def test_export_pytorch_package_allows_reshape_only_residual_layout_bridge(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_only_residual_layout_bridge")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 8, 6, 10],
        shape_signature=[1, 8, 6, 10],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_bridge"] = TensorIR(
        name="x_bridge",
        dtype="FLOAT32",
        shape=[1, 8, 6, 10],
        shape_signature=[1, 8, 6, 10],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 8, 2, 3, 2, 5],
        shape_signature=[1, 8, 2, 3, 2, 5],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["x_bridge"], options={}),
            OperatorIR(op_type="RESHAPE", inputs=["x_bridge"], outputs=["y"], options={"newShape": [1, 8, 2, 3, 2, 5]}),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_only_residual_layout_bridge_pkg"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("execution_backend") == "native"
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "x_bridge = x.permute" not in model_py


def test_normalize_model_ir_rewrites_shared_reduce_axes_only_once() -> None:
    model_ir = ModelIR(name="shared_reduce_axes_once")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["mean0", "mean1"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 5, 3],
        shape_signature=[1, 4, 5, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 2], dtype=np.int32),
    )
    model_ir.tensors["mean0"] = TensorIR(
        name="mean0",
        dtype="FLOAT32",
        shape=[1, 1, 1, 3],
        shape_signature=[1, 1, 1, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["mean1"] = TensorIR(
        name="mean1",
        dtype="FLOAT32",
        shape=[1, 1, 1, 3],
        shape_signature=[1, 1, 1, 3],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="MEAN", inputs=["x", "axes"], outputs=["mean0"], options={"keepDims": True}),
            OperatorIR(op_type="MEAN", inputs=["x", "axes"], outputs=["mean1"], options={"keepDims": True}),
        ]
    )

    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)
    axes_tensor = normalized.tensors["axes"]
    assert np.array_equal(np.asarray(axes_tensor.data), np.asarray([2, 3], dtype=np.int32))


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
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "def _device" not in model_source
    assert "    _module_device,\n" in model_source
    assert "device=_module_device(self)" in model_source
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


def test_build_torchscript_example_inputs_autoresolves_batch_only_dynamic_input(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_batch_only_trace_hint")
    model_ir.inputs = ["dense_input__0"]
    model_ir.outputs = ["y"]
    model_ir.tensors["dense_input__0"] = TensorIR(
        name="dense_input__0",
        dtype="FLOAT32",
        shape=[1, 784],
        shape_signature=[-1, 784],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 784],
        shape_signature=[-1, 784],
    )
    model_ir.operators.append(
        OperatorIR(op_type="ABS", inputs=["dense_input__0"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_batch_only_pkg"),
    )
    with open(Path(package_path) / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    example_inputs, example_input_shapes, dynamic_inputs_present = _build_torchscript_example_inputs(
        package_dir=str(package_path),
        package_metadata=metadata,
        custom_input_op_name_np_data_path=None,
        shape_hints=None,
        test_data_nhwc_path=None,
    )

    assert dynamic_inputs_present is True
    assert len(example_inputs) == 1
    assert list(example_inputs[0].shape) == [1, 784]
    assert example_input_shapes == {"dense_input__0": [1, 784]}


def test_build_torchscript_example_inputs_still_requires_hint_for_non_batch_dynamic_input(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_feature_trace_hint")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, -1],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, -1],
    )
    model_ir.operators.append(
        OperatorIR(op_type="ABS", inputs=["x"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_feature_pkg"),
    )
    with open(Path(package_path) / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with pytest.raises(ModelIRPyTorchExportError, match="requires concrete trace hints"):
        _build_torchscript_example_inputs(
            package_dir=str(package_path),
            package_metadata=metadata,
            custom_input_op_name_np_data_path=None,
            shape_hints=None,
            test_data_nhwc_path=None,
        )


def test_export_dynamo_onnx_uses_shape_hints_for_dynamic_inputs(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_feature_dynamo_onnx")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, -1],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, -1],
    )
    model_ir.operators.append(
        OperatorIR(op_type="ABS", inputs=["x"], outputs=["y"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_feature_dynamo_pkg"),
    )

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        shape_hints=["x:1,4"],
        raise_on_failure=False,
    )

    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dynamo_onnx"]["example_input_shapes"] == {"x": [1, 4]}
    assert metadata["dynamo_onnx"]["dynamic_inputs_present"] is True


def test_exported_program_records_failure_for_missing_dynamic_input_hint(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_feature_exported_program")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, -1],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, -1],
    )
    model_ir.operators.append(
        OperatorIR(op_type="ABS", inputs=["x"], outputs=["y"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_feature_exported_program_pkg"),
    )

    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        raise_on_failure=False,
    )

    assert exported_program_path is None
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["exported_program"]["file_name"] is None
    assert metadata["exported_program"]["dynamic_inputs_present"] is True
    assert "requires concrete trace hints" in metadata["exported_program"]["error"]


def test_export_pytorch_package_slice_with_dynamic_tail_dim_keeps_finite_stop(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_tail_slice_codegen")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 49, 8, -1],
        shape_signature=[1, 49, 8, -1],
    )
    model_ir.tensors["begin"] = TensorIR(
        name="begin",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 0, 0, 32], dtype=np.int32),
    )
    model_ir.tensors["size"] = TensorIR(
        name="size",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 49, 8, 32], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 49, 8, 32],
        shape_signature=[1, 49, 8, 32],
    )
    model_ir.operators.append(
        OperatorIR(op_type="SLICE", inputs=["x", "begin", "size"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_tail_slice_codegen_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "32:64" in model_source
    assert "32:-1" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 49 * 8 * 192, dtype=torch.float32).reshape(1, 49, 8, 192)
    out = cast(Any, model).forward_named(x=x)["y"]
    assert torch.equal(out, x[:, :, :, 32:64])


def test_export_if_axis0_tensor_mux_supports_dynamo_onnx_and_exported_program(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_if_axis0_tensor_mux_model_ir(),
        output_folder_path=str(tmp_path / "if_axis0_tensor_mux_pkg"),
    )

    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_apply_if_axis0_tensor_mux(" in model_source
    assert "self._onnx2tf_torch_export_mode = False" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x1_true = torch.full((1, 100), 3.0, dtype=torch.float32)
    x2_true = torch.zeros((2, 100), dtype=torch.float32)
    y_true = cast(Any, model).forward_named(If_p1_input1=x1_true, If_p1_input2=x2_true)["If_p1_output"]
    assert y_true.shape == (1, 100)
    assert torch.allclose(y_true, x1_true + 1.0)

    x1_false = torch.zeros((1, 100), dtype=torch.float32)
    x2_false = torch.full((2, 100), 1.0, dtype=torch.float32)
    y_false = cast(Any, model).forward_named(If_p1_input1=x1_false, If_p1_input2=x2_false)["If_p1_output"]
    assert y_false.shape == (2, 100)
    assert torch.allclose(y_false, x2_false + 2.0)

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)

    assert torchscript_path is not None
    assert dynamo_onnx_path is not None
    assert exported_program_path is not None
    assert Path(torchscript_path).exists()
    assert Path(dynamo_onnx_path).exists()
    assert Path(exported_program_path).exists()

    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["torchscript"]["file_name"] is not None
    assert metadata["dynamo_onnx"]["file_name"] is not None
    assert metadata["exported_program"]["file_name"] is not None


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


def test_export_pytorch_package_codegen_supports_multidim_constant_gather_indices(tmp_path) -> None:
    model_ir = ModelIR(name="gather_multidim_constant_indices")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[2, 2, 5, 3],
        shape_signature=[2, 2, 5, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[2, 2],
        shape_signature=[2, 2],
        data=np.asarray([[0, 1], [2, 4]], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[2, 2, 2, 2, 3],
        shape_signature=[2, 2, 2, 2, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(op_type="GATHER", inputs=["x", "indices"], outputs=["y"], options={"axis": 2, "batchDims": 0})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "gather_multidim_constant_indices_pytorch"),
    )
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.index_select(" in model_py
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(2 * 2 * 5 * 3, dtype=torch.float32).reshape(2, 2, 5, 3)
    out = model(x)
    expected = torch.index_select(x, 2, torch.tensor([0, 1, 2, 4], dtype=torch.int64)).reshape(2, 2, 2, 2, 3)
    assert torch.allclose(out, expected)


def test_export_pytorch_package_codegen_supports_scalar_constant_gather_indices(tmp_path) -> None:
    model_ir = ModelIR(name="gather_scalar_constant_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[2, 1, 3, 4],
        shape_signature=[2, 1, 3, 4],
    )
    model_ir.tensors["index"] = TensorIR(
        name="index",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(0, dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[2, 3, 4],
        shape_signature=[2, 3, 4],
    )
    model_ir.operators.append(
        OperatorIR(op_type="GATHER", inputs=["x", "index"], outputs=["y"], options={"axis": 1, "batchDims": 0})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "gather_scalar_constant_index_pytorch"),
    )
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".squeeze(1)" in model_py
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)
    out = model(x)
    expected = torch.index_select(x, 1, torch.tensor([0], dtype=torch.int64)).squeeze(1)
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
    assert "F.interpolate(" in model_py or "_apply_resize(" in model_py
    assert "torch.cat(" in model_py
    assert "_align_tensor_to_target_shape(" not in model_py
    assert "up_nhwc =" not in model_py
    assert "y_nhwc =" not in model_py
    assert ".permute(0, 2, 3, 1).contiguous()" not in model_py
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    permute_nodes = [
        node
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.permute.default"
    ]
    assert permute_nodes == []
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


def test_direct_codegen_concat_aligns_layout_to_target_shape(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_concat_with_layout_alignment_model_ir(),
        output_folder_path=str(tmp_path / "concat_layout_alignment_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 1 * 4 * 3), dtype=torch.float32).reshape(1, 1, 4, 3)
    y = x + 100.0
    out = model(x, y)
    ref = torch.cat(
        [
            x.permute(0, 3, 1, 2).contiguous(),
            y.permute(0, 3, 1, 2).contiguous(),
        ],
        dim=2,
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_align_tensor_to_target_shape_narrows_oversized_static_dims() -> None:
    value = torch.arange(1 * 1 * 81 * 512, dtype=torch.float32).reshape(1, 1, 81, 512)
    out = _align_tensor_to_target_shape(value, [1, 1, 40, 512])
    assert tuple(out.shape) == (1, 1, 40, 512)
    assert torch.equal(out, value[:, :, :40, :])


def test_export_artifacts_unroll_static_while_sigmoid_chain(tmp_path) -> None:
    model_ir = _make_static_sigmoid_while_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "static_sigmoid_while_pkg"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "native"
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "WHILE" not in model_source
    assert model_source.count("torch.sigmoid(") >= 8

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([0.25, -1.0, 2.0], dtype=torch.float32)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = x
    for _ in range(8):
        expected = torch.sigmoid(expected)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_unroll_counter_bounded_while_sigmoid_chain(tmp_path) -> None:
    model_ir = _make_counter_bounded_sigmoid_while_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "counter_bounded_sigmoid_while_pkg"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "native"
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "WHILE" not in model_source
    assert "torch.where(" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([0.25, -1.0, 2.0], dtype=torch.float32)
    counter = torch.tensor([3], dtype=torch.int64)
    out = cast(Any, model).forward_named(x=x, counter=counter)["y"]
    expected = x
    for _ in range(61):
        expected = torch.sigmoid(expected)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_unroll_counter_bounded_select_while_sigmoid_chain(tmp_path) -> None:
    model_ir = _make_counter_bounded_select_sigmoid_while_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "counter_bounded_select_sigmoid_while_pkg"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "native"
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "WHILE" not in model_source
    assert "torch.where(" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([0.25, -1.0, 2.0], dtype=torch.float32)
    counter = torch.tensor([3], dtype=torch.int64)
    out = cast(Any, model).forward_named(x=x, counter=counter)["y"]
    expected = x
    for _ in range(61):
        expected = torch.sigmoid(expected)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_direct_codegen_concat_keeps_axis_only_match_for_ambiguous_square_tensor(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_concat_with_ambiguous_axis_only_match_model_ir(),
        output_folder_path=str(tmp_path / "concat_ambiguous_axis_only_match_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    boxes = torch.arange(1, 1 + (1 * 4 * 8 * 8), dtype=torch.float32).reshape(1, 4, 8, 8)
    obj = torch.arange(1000, 1000 + (1 * 1 * 8 * 8), dtype=torch.float32).reshape(1, 1, 8, 8)
    cls = torch.arange(2000, 2000 + (1 * 8 * 8 * 8), dtype=torch.float32).reshape(1, 8, 8, 8)
    out = model(boxes, obj, cls)
    expected = torch.cat([boxes, obj, cls], dim=1)
    assert torch.equal(out, expected)


def test_direct_codegen_concat_keeps_rank3_dynamic_middle_dim_orientation(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_concat_with_rank3_dynamic_middle_dim_model_ir(),
        output_folder_path=str(tmp_path / "concat_rank3_dynamic_middle_dim_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    boxes = torch.arange(1, 1 + (1 * 34 * 4), dtype=torch.float32).reshape(1, 34, 4)
    scores = torch.arange(1000, 1000 + (1 * 34 * 1), dtype=torch.float32).reshape(1, 34, 1)
    classes = torch.arange(2000, 2000 + (1 * 34 * 1), dtype=torch.float32).reshape(1, 34, 1)
    out = model(boxes, scores, classes)
    expected = torch.cat([boxes, scores, classes], dim=2)
    assert torch.equal(out, expected)


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


def test_remove_redundant_layout_transposes_keeps_non_layout_shape_transpose() -> None:
    model_ir = _make_non_layout_shape_transpose_model_ir()
    _remove_redundant_layout_transposes(
        model_ir,
        original_layouts={"x": "NCW", "y": "NCW"},
        preserve_channel_last_tensor_names=set(),
    )
    assert model_ir.operators[0].op_type == "TRANSPOSE"


def test_validate_channel_first_exportability_allows_degenerate_unknown_layout_slice() -> None:
    validate_channel_first_exportability(
        _make_degenerate_unknown_layout_slice_model_ir(),
        preserve_channel_last_tensor_names=set(),
    )


def test_validate_channel_first_exportability_ignores_scatter_nd_indices_layout() -> None:
    validate_channel_first_exportability(
        _make_scatter_nd_indices_unknown_layout_model_ir(),
        preserve_channel_last_tensor_names=set(),
    )


def test_validate_channel_first_exportability_ignores_scatter_nd_updates_layout() -> None:
    validate_channel_first_exportability(
        _make_scatter_nd_updates_unknown_layout_model_ir(),
        preserve_channel_last_tensor_names=set(),
    )


def test_validate_channel_first_exportability_ignores_internal_scatter_nd_output_layout() -> None:
    validate_channel_first_exportability(
        _make_scatter_nd_internal_output_unknown_layout_model_ir(),
        preserve_channel_last_tensor_names=set(),
    )


def test_infer_model_ir_logical_layouts_uses_nwc_for_recurrent_public_rank3_boundaries() -> None:
    model_ir = _make_recurrent_public_boundary_model_ir()
    infer_model_ir_logical_layouts(model_ir)
    assert model_ir.metadata["onnx_public_layout_map"]["data_input"] == "NWC"
    assert model_ir.metadata["onnx_public_layout_map"]["preds"] == "NWC"


def test_propagate_pytorch_friendly_layouts_infers_unknown_concat_peer_from_known_branch() -> None:
    model_ir = _make_concat_unknown_peer_layout_model_ir()
    _propagate_pytorch_friendly_layouts(model_ir)
    assert normalize_logical_layout(model_ir.tensors["branch_unknown"].logical_layout) == "NCHW"
    assert normalize_logical_layout(model_ir.tensors["y"].logical_layout) == "NCHW"


def test_infer_model_ir_logical_layouts_detects_nhwc_public_input_from_boundary_transpose_chain() -> None:
    model_ir = _make_nhwc_public_input_boundary_model_ir()
    infer_model_ir_logical_layouts(model_ir)
    assert model_ir.metadata["onnx_public_layout_map"]["image"] == "NHWC"
    assert model_ir.tensors["image"].logical_layout == "NHWC"


def test_infer_model_ir_logical_layouts_detects_nhwc_public_output_from_boundary_transpose_chain() -> None:
    model_ir = _make_nhwc_public_output_boundary_model_ir()
    infer_model_ir_logical_layouts(model_ir)
    assert model_ir.metadata["onnx_public_layout_map"]["image"] == "NHWC"


def test_export_pytorch_package_prefers_native_package_for_supported_channel_first_softmax(
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
    assert metadata.get("execution_backend") is None
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "load_generated_model_package" not in model_py


def test_should_prefer_tflite_backend_for_large_detection_head_runtime_fallback() -> None:
    model_ir = _make_large_detection_head_model_ir()
    assert _should_prefer_tflite_backed_package(model_ir) is True


def test_should_prefer_tflite_backend_for_large_nhwc_heavy_graph() -> None:
    model_ir = _make_large_nhwc_heavy_model_ir()
    assert _should_prefer_tflite_backed_package(model_ir) is True


def test_export_pytorch_package_does_not_force_tflite_backend_for_large_detection_head_when_native_is_possible(
    tmp_path,
) -> None:
    model_ir = _make_large_detection_head_model_ir()
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "large_detection_head", model_ir)
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "large_detection_head_pytorch"),
        fallback_tflite_path=tflite_path,
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("execution_backend") != "tflite"


def test_native_nms_postprocess_chain_keeps_public_selected_indices_and_crops_internal_consumers_when_empty(
    tmp_path,
) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_nms_postprocess_chain_model_ir(score_threshold=0.99),
        output_folder_path=str(tmp_path / "nms_postprocess_chain_empty_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [10.0, 10.0, 11.0, 11.0], [20.0, 20.0, 21.0, 21.0], [30.0, 30.0, 31.0, 31.0]]],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.95, 0.90, 0.70, 0.20, 0.80]]], dtype=torch.float32)
    outputs = cast(Any, model).forward_named(boxes=boxes, scores=scores)
    assert outputs["selected_indices"].tolist() == [0, 0, 0]
    assert int(outputs["valid_count"].item()) == 0
    assert tuple(outputs["head"].shape) == (1, 0, 5)


def test_native_nms_postprocess_chain_handles_positive_valid_count_and_exports_artifacts(
    tmp_path,
) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_nms_postprocess_chain_model_ir(score_threshold=0.4),
        output_folder_path=str(tmp_path / "nms_postprocess_chain_positive_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [10.0, 10.0, 11.0, 11.0], [20.0, 20.0, 21.0, 21.0], [30.0, 30.0, 31.0, 31.0]]],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.95, 0.90, 0.70, 0.20, 0.80]]], dtype=torch.float32)
    outputs = cast(Any, model).forward_named(boxes=boxes, scores=scores)
    assert outputs["selected_indices"].tolist() == [0, 4, 2]
    assert int(outputs["valid_count"].item()) == 3
    expected_head = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0, 0.95], [30.0, 30.0, 31.0, 31.0, 0.80], [10.0, 10.0, 11.0, 11.0, 0.70]]],
        dtype=torch.float32,
    )
    assert torch.allclose(outputs["head"], expected_head)

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


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


def test_export_pytorch_package_prefers_reimported_native_package_over_saved_model_fallback(
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
    assert metadata.get("execution_backend") is None
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "load_generated_model_package" not in model_py


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


def test_native_softmax_codegen_respects_explicit_last_axis_on_ncw_tensor(tmp_path) -> None:
    model_ir = ModelIR(name="softmax_native_codegen_last_axis")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[8, 225, 225],
        shape_signature=[8, 225, 225],
        logical_layout="NCW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[8, 225, 225],
        shape_signature=[8, 225, 225],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SOFTMAX",
            inputs=["x"],
            outputs=["y"],
            options={"axis": 2, "beta": 1.0},
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "softmax_native_codegen_last_axis_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_apply_softmax(x, axis=2, beta=1.0" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.Model(load_weights=True, eval_mode=True)
    x = torch.arange(8 * 225 * 225, dtype=torch.float32).reshape(8, 225, 225) / 1000.0
    out = model(x)
    ref = torch.softmax(x, dim=2)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


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


def test_export_generated_package_resizes_channel_last_tensor_to_channel_first_public_output(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="channel_last_resize_to_channel_first_output")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 5, 3],
        shape_signature=[1, 4, 5, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["size"] = TensorIR(
        name="size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([8, 9], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 8, 9],
        shape_signature=[1, 3, 8, 9],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESIZE_BILINEAR",
            inputs=["x", "size"],
            outputs=["y"],
            options={
                "alignCorners": True,
                "halfPixelCenters": False,
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "channel_last_resize_to_channel_first_output_pkg"),
    )

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 4 * 5 * 3, dtype=torch.float32).reshape(1, 4, 5, 3) / 10.0
    with torch.no_grad():
        y = model(x)

    expected = F.interpolate(
        x.permute(0, 3, 1, 2).contiguous(),
        size=(8, 9),
        mode="bilinear",
        align_corners=True,
    )
    assert y.shape == (1, 3, 8, 9)
    assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5)


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


def test_export_pytorch_package_elides_crd_to_dcr_reorder_for_native_depth_to_space(tmp_path) -> None:
    model_ir = ModelIR(name="native_crd_depth_to_space")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors["input"] = TensorIR(
        name="input",
        dtype="FLOAT32",
        shape=[2, 12, 8, 8],
        shape_signature=[2, 12, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["wa/DepthToSpace_crd_to_dcr_indices"] = TensorIR(
        name="wa/DepthToSpace_crd_to_dcr_indices",
        dtype="INT32",
        shape=[12],
        shape_signature=[12],
        data=np.asarray([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["reordered"] = TensorIR(
        name="reordered",
        dtype="FLOAT32",
        shape=[2, 12, 8, 8],
        shape_signature=[2, 12, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[2, 3, 16, 16],
        shape_signature=[2, 3, 16, 16],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="GATHER",
            inputs=["input", "wa/DepthToSpace_crd_to_dcr_indices"],
            outputs=["reordered"],
            options={"axis": 1, "batchDims": 0},
        )
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="DEPTH_TO_SPACE",
            inputs=["reordered"],
            outputs=["output"],
            options={"blockSize": 2},
        )
    )
    infer_model_ir_logical_layouts(model_ir)

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "native_crd_depth_to_space_pkg"),
    )
    package_dir = Path(package_path)
    model_source = (package_dir / "model.py").read_text(encoding="utf-8")
    assert "_crd_to_dcr_indices" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(2 * 12 * 8 * 8, dtype=torch.float32).reshape(2, 12, 8, 8)
    out = cast(Any, model).forward_named(input=x)["output"]
    expected = F.pixel_shuffle(x, 2)
    assert torch.allclose(out, expected)


def test_export_pytorch_package_reshape_allow_zero_uses_input_dims(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_allow_zero_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 6],
        shape_signature=[1, 4, 6],
        logical_layout="NCW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 2, -1], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 12],
        shape_signature=[1, 2, 12],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={
                "newShape": [1, 2, 12],
                "onnxRawNewShape": [0, 2, -1],
                "allowZero": False,
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_allow_zero_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = torch.reshape(x, [1, 2, 12])
    assert torch.equal(out, expected)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_handles_local_response_normalization(tmp_path) -> None:
    model_ir = ModelIR(name="lrn_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 5, 5],
        shape_signature=[1, 3, 5, 5],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 5, 5],
        shape_signature=[1, 3, 5, 5],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="LOCAL_RESPONSE_NORMALIZATION",
            inputs=["x"],
            outputs=["y"],
            options={
                "radius": 2,
                "alpha": 2e-5,
                "beta": 0.75,
                "bias": 1.0,
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "lrn_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(75, dtype=torch.float32).reshape(1, 3, 5, 5)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = F.local_response_norm(x, size=5, alpha=2e-5, beta=0.75, k=1.0)
    assert torch.allclose(out, expected)


def test_export_pytorch_package_handles_scatter_nd_with_unknown_updates_layout(tmp_path) -> None:
    model_ir = _make_scatter_nd_updates_unknown_layout_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "scatter_nd_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    updates = torch.tensor([[[3.0, 7.0]]], dtype=torch.float32)
    out = cast(Any, model).forward_named(updates=updates)["y"]
    expected = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    expected[0, 0, 0, 1] = 3.0
    expected[0, 0, 1, 2] = 7.0
    assert torch.equal(out, expected)


def test_export_pytorch_package_handles_internal_scatter_nd_output_unknown_layout(tmp_path) -> None:
    model_ir = _make_scatter_nd_internal_output_unknown_layout_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "scatter_nd_internal_output_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    data = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    out = cast(Any, model).forward_named(data=data)["y"]
    expected = data.clone()
    expected[0, 0, 0, 1] += 3.0
    expected[0, 0, 1, 2] += 7.0
    assert torch.equal(out, expected)


def test_export_pytorch_package_handles_scatter_nd_channel_first_updates_layout(tmp_path) -> None:
    model_ir = _make_scatter_nd_channel_first_updates_layout_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "scatter_nd_channel_first_updates_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    updates = torch.arange(1 * 4 * 1 * 2 * 3, dtype=torch.float32).reshape(1, 4, 1, 2, 3)
    out = cast(Any, model).forward_named(updates=updates)["y"]
    assert torch.equal(out, updates)


def test_export_pytorch_package_runtime_align_scatter_nd_updates_permutates_square_channel_first_case(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_model_ir(),
        output_folder_path=str(tmp_path / "scatter_nd_runtime_square_pkg"),
    )
    parent = str(Path(package_path).parent)
    package_name = Path(package_path).name
    sys.path.insert(0, parent)
    try:
        runtime = importlib.import_module(f"{package_name}.runtime")
    finally:
        if sys.path and sys.path[0] == parent:
            sys.path.pop(0)
    updates = torch.arange(1 * 4 * 1 * 1 * 4, dtype=torch.float32).reshape(1, 4, 1, 1, 4)
    aligned = runtime._align_scatter_nd_updates(updates, [1, 1, 1, 4, 4])
    expected = updates.permute(0, 2, 3, 4, 1).contiguous()
    assert torch.equal(aligned, expected)


def test_export_torchscript_handles_scatter_nd_channel_first_updates_layout(tmp_path) -> None:
    model_ir = _make_scatter_nd_channel_first_updates_layout_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "scatter_nd_channel_first_updates_torchscript_pkg"),
    )
    torchscript_path = export_torchscript_from_generated_package(
        package_dir=package_path,
        raise_on_failure=False,
    )
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()

    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["torchscript"]["trace_mode"] in {"trace", "script"}


def test_export_pytorch_package_handles_gather_elements_axis_coord_unsqueeze(tmp_path) -> None:
    model_ir = _make_gather_elements_axis_coord_unsqueeze_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "gather_elements_axis_coord_unsqueeze_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    out = cast(Any, model).forward_named(x=x)["wa/GatherElements_output_0_gather_elements_axis_coord"]
    expected = torch.topk(x, k=2, dim=-1, largest=True, sorted=True).indices.to(dtype=torch.int32).unsqueeze(-1)
    assert torch.equal(out, expected)


def test_export_pytorch_package_handles_gather_elements_dynamic_coords(tmp_path) -> None:
    model_ir = _make_gather_elements_dynamic_coords_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "gather_elements_dynamic_coords_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = torch.topk(x, k=2, dim=-1, largest=True, sorted=True).values
    assert torch.equal(out, expected)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)

    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_handles_topk_layout_bridge_with_indices_axis_restore(tmp_path) -> None:
    model_ir = ModelIR(name="topk_layout_bridge_indices_restore_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["indices"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 5, 7],
        shape_signature=[1, 3, 5, 7],
        logical_layout="NCHW",
    )
    model_ir.tensors["k"] = TensorIR(
        name="k",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(3, dtype=np.int32),
    )
    model_ir.tensors["values_raw"] = TensorIR(
        name="values_raw",
        dtype="FLOAT32",
        shape=[1, 5, 7, 3],
        shape_signature=[1, 5, 7, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["indices_raw"] = TensorIR(
        name="indices_raw",
        dtype="INT32",
        shape=[1, 3, 5, 7],
        shape_signature=[1, 3, 5, 7],
        logical_layout="NCHW",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT64",
        shape=[1, 3, 5, 7],
        shape_signature=[1, 3, 5, 7],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="TOPK_V2",
                inputs=["x", "k"],
                outputs=["values_raw", "indices_raw"],
                options={},
            ),
            OperatorIR(
                op_type="CAST",
                inputs=["indices_raw"],
                outputs=["indices"],
                options={"inDataType": "INT32", "outDataType": "INT64"},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "topk_layout_bridge_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7)
    out = cast(Any, model).forward_named(x=x)["indices"]
    expected = torch.topk(x.permute(0, 2, 3, 1), k=3, dim=-1, largest=True, sorted=True).indices.permute(0, 3, 1, 2).to(dtype=torch.int64)
    assert torch.equal(out, expected)


def test_export_pytorch_package_uses_sort_for_dynamic_full_length_topk_and_keeps_empty_gather_valid(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_dynamic_full_sort_topk_model_ir(),
        output_folder_path=str(tmp_path / "dynamic_full_sort_topk_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.sort(" in model_source
    assert "torch.topk(" not in model_source
    assert "_shape_tensor(sorted_indices" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()

    scores = torch.tensor([0.25, 0.9, 0.1, 0.7], dtype=torch.float32)
    positions = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    outputs = cast(Any, model).forward_named(scores=scores, positions=positions)
    expected_values, expected_indices = torch.sort(scores, descending=True)
    expected_positions = positions.index_select(0, expected_indices.to(dtype=torch.int64))
    assert torch.allclose(outputs["sorted_values"], expected_values)
    assert torch.equal(outputs["sorted_indices"], expected_indices.to(dtype=torch.int32))
    assert torch.equal(outputs["sorted_positions"], expected_positions)

    empty_outputs = cast(Any, model).forward_named(
        scores=torch.empty([0], dtype=torch.float32),
        positions=torch.empty([0], dtype=torch.int32),
    )
    assert tuple(empty_outputs["sorted_values"].shape) == (0,)
    assert tuple(empty_outputs["sorted_indices"].shape) == (0,)
    assert tuple(empty_outputs["sorted_positions"].shape) == (0,)

    torchscript_path = export_torchscript_from_generated_package(
        package_dir=package_path,
        raise_on_failure=False,
    )
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()


def test_export_pytorch_package_handles_compare_and_logical_binary_ops(tmp_path) -> None:
    model_ir = ModelIR(name="compare_and_logical_binary_ops_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["floor_mod", "greater", "equal", "logical_and", "logical_or"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT32",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="INT32",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["floor_mod"] = TensorIR(
        name="floor_mod",
        dtype="INT32",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["greater"] = TensorIR(
        name="greater",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["equal"] = TensorIR(
        name="equal",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["logical_and"] = TensorIR(
        name="logical_and",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["logical_or"] = TensorIR(
        name="logical_or",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["not_equal"] = TensorIR(
        name="not_equal",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="FLOOR_MOD", inputs=["x", "y"], outputs=["floor_mod"], options={}),
            OperatorIR(op_type="GREATER", inputs=["x", "y"], outputs=["greater"], options={}),
            OperatorIR(op_type="EQUAL", inputs=["x", "y"], outputs=["equal"], options={}),
            OperatorIR(op_type="NOT_EQUAL", inputs=["x", "y"], outputs=["not_equal"], options={}),
            OperatorIR(op_type="LOGICAL_AND", inputs=["greater", "equal"], outputs=["logical_and"], options={}),
            OperatorIR(op_type="LOGICAL_OR", inputs=["greater", "equal"], outputs=["logical_or"], options={}),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "compare_logic_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[5, -3, 4], [6, 8, -7]], dtype=torch.int32)
    y = torch.tensor([[2, 2, -3], [4, 8, 3]], dtype=torch.int32)
    out = cast(Any, model).forward_named(x=x, y=y)
    assert torch.equal(out["floor_mod"], torch.remainder(x, y))
    assert torch.equal(out["greater"], torch.gt(x, y))
    assert torch.equal(out["equal"], torch.eq(x, y))
    assert torch.equal(out["not_equal"], torch.ne(x, y))
    assert torch.equal(out["logical_and"], torch.logical_and(torch.gt(x, y), torch.eq(x, y)))
    assert torch.equal(out["logical_or"], torch.logical_or(torch.gt(x, y), torch.eq(x, y)))


def test_export_torchscript_handles_bool_scalar_compare_ops(tmp_path) -> None:
    model_ir = ModelIR(name="bool_scalar_compare_trace")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["equal_false", "not_equal_false"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["false_scalar"] = TensorIR(
        name="false_scalar",
        dtype="BOOL",
        shape=[],
        shape_signature=[],
        data=np.asarray(False, dtype=np.bool_),
    )
    model_ir.tensors["equal_false"] = TensorIR(
        name="equal_false",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.tensors["not_equal_false"] = TensorIR(
        name="not_equal_false",
        dtype="BOOL",
        shape=[2, 3],
        shape_signature=[2, 3],
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="EQUAL", inputs=["x", "false_scalar"], outputs=["equal_false"], options={}),
            OperatorIR(op_type="NOT_EQUAL", inputs=["x", "false_scalar"], outputs=["not_equal_false"], options={}),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "bool_scalar_compare_trace_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.as_tensor(False, dtype=x.dtype, device=x.device)" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[True, False, True], [False, False, True]], dtype=torch.bool)
    out = cast(Any, model).forward_named(x=x)
    assert torch.equal(out["equal_false"], torch.eq(x, False))
    assert torch.equal(out["not_equal_false"], torch.ne(x, False))

    torchscript_path = export_torchscript_from_generated_package(
        package_dir=package_path,
        raise_on_failure=False,
    )
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()

    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["torchscript"]["file_name"] is not None
    assert metadata["torchscript"]["trace_mode"] == "trace"


def test_export_pytorch_package_aligns_dynamic_binary_inputs_with_ambiguous_static_shapes(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_binary_alignment_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[-1, -1, -1, -1],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[1, 1, -1, 1],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[-1, -1, -1, -1],
    )
    model_ir.operators.append(
        OperatorIR(op_type="MUL", inputs=["x", "y"], outputs=["z"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_binary_alignment_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 5)
    y = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
    out = cast(Any, model).forward_named(x=x, y=y)["z"]
    expected = x * y.permute(0, 1, 3, 2).contiguous()
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_export_pytorch_package_prefers_runtime_shape_input_for_ambiguous_reshape_output(tmp_path) -> None:
    model_ir = ModelIR(name="runtime_shape_input_reshape_model_ir")
    model_ir.inputs = ["x", "shape"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, -1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[-1, -1, -1, -1],
    )
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "runtime_shape_input_reshape_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    shape = torch.tensor([1, 1, 2, 3], dtype=torch.int32)
    out = cast(Any, model).forward_named(x=x, shape=shape)["y"]
    expected = x.reshape(1, 1, 2, 3)
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_generated_runtime_align_binary_inputs_prefers_permutation_over_ambiguous_broadcast(tmp_path) -> None:
    model_ir = ModelIR(name="binary_align_runtime_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.operators.append(
        OperatorIR(op_type="MUL", inputs=["x", "y"], outputs=["z"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "binary_align_runtime_pkg"),
    )
    package_name = Path(package_path).name
    pkg = _import_generated_package(package_path)
    runtime = importlib.import_module(f"{package_name}.runtime")
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 6)
    y = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
    aligned_x, aligned_y = runtime._align_binary_inputs(x, y, [1, 1, 1, 1])
    assert tuple(aligned_x.shape) == (1, 1, 1, 6)
    assert tuple(aligned_y.shape) == (1, 1, 1, 6)
    assert torch.equal(aligned_y, y.permute(0, 1, 3, 2).contiguous())
    del pkg


def test_generated_runtime_align_binary_inputs_keeps_broadcastable_constants_for_dynamic_targets(tmp_path) -> None:
    model_ir = ModelIR(name="binary_align_runtime_dynamic_target_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.operators.append(
        OperatorIR(op_type="MUL", inputs=["x", "y"], outputs=["z"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "binary_align_runtime_dynamic_target_pkg"),
    )
    package_name = Path(package_path).name
    pkg = _import_generated_package(package_path)
    runtime = importlib.import_module(f"{package_name}.runtime")
    x = torch.arange(1 * 511 * 96 * 2, dtype=torch.float32).reshape(1, 511, 96, 2)
    y = torch.tensor([[[[0.25, -0.75]]]], dtype=torch.float32)
    aligned_x, aligned_y = runtime._align_binary_inputs(x, y, [-1, 511, 96, 2])
    assert tuple(aligned_x.shape) == (1, 511, 96, 2)
    assert tuple(aligned_y.shape) == (1, 1, 1, 2)
    assert torch.equal(aligned_y, y)
    del pkg


def test_scatter_nd_with_constant_shape_supports_dynamo_onnx_and_exported_program(tmp_path) -> None:
    model_ir = ModelIR(name="scatter_nd_const_shape_export")
    model_ir.inputs = ["updates"]
    model_ir.outputs = ["y"]
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1, 2, 4],
        shape_signature=[1, 1, 2, 4],
        data=np.asarray([[[[0, 0, 0, 1], [0, 0, 1, 2]]]], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["updates"] = TensorIR(
        name="updates",
        dtype="FLOAT32",
        shape=[1, 1, 2],
        shape_signature=[1, 1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 2, 3], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[1, 1, 2, 3],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=["indices", "updates", "shape"],
            outputs=["y"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "scatter_nd_const_shape_export_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_apply_scatter_nd(self.const_indices, updates, [1, 1, 2, 3]" in model_source

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)

    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_scatter_nd_with_dynamic_prefix_supports_dynamo_onnx_and_exported_program(tmp_path) -> None:
    model_ir = ModelIR(name="scatter_nd_dynamic_prefix_export")
    model_ir.inputs = ["indices", "updates"]
    model_ir.outputs = ["y"]
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["updates"] = TensorIR(
        name="updates",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([5], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[5],
        shape_signature=[5],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=["indices", "updates", "shape"],
            outputs=["y"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "scatter_nd_dynamic_prefix_export_pkg"),
    )
    runtime_source = (Path(package_path) / "runtime.py").read_text(encoding="utf-8")
    assert "prefix_shape = list(indices_i64.shape[:-1])" in runtime_source
    assert "prefix_shape = [int(v) for v in list(indices_i64.shape[:-1])]" not in runtime_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    indices = torch.tensor([[0], [2], [4]], dtype=torch.int32)
    updates = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    out = cast(Any, model).forward_named(indices=indices, updates=updates)["y"]
    assert torch.equal(out, torch.tensor([1.5, 0.0, 2.5, 0.0, 3.5], dtype=torch.float32))

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        shape_hints=["indices:3,1", "updates:3"],
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        shape_hints=["indices:3,1", "updates:3"],
    )

    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_uses_truncating_integer_division(tmp_path) -> None:
    model_ir = ModelIR(name="integer_div_codegen")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT32",
        shape=[1, 1, 1, 4],
        shape_signature=[1, 1, 1, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["two"] = TensorIR(
        name="two",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="INT32",
        shape=[1, 1, 1, 4],
        shape_signature=[1, 1, 1, 4],
        logical_layout="NHWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="DIV",
            inputs=["x", "two"],
            outputs=["y"],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "integer_div_codegen_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "rounding_mode='trunc'" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32)
    with torch.no_grad():
        out = model(x)

    expected = torch.tensor([[[[0, 0, 1, 1]]]], dtype=torch.int32)
    assert torch.equal(out, expected)


def test_export_pytorch_package_uses_shape_signature_for_dynamic_gather_nd_targets(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_gather_nd_target_shape")
    model_ir.inputs = ["params", "indices"]
    model_ir.outputs = ["y"]
    model_ir.tensors["params"] = TensorIR(
        name="params",
        dtype="FLOAT32",
        shape=[5, 2],
        shape_signature=[5, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[-1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=["params", "indices"],
            outputs=["y"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_gather_nd_target_shape_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "target_shape=[-1, 2]" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    params = torch.tensor(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[0], [2], [4]], dtype=torch.int32)
    out = cast(Any, model).forward_named(params=params, indices=indices)["y"]
    expected = params[indices[:, 0].to(dtype=torch.int64)]
    assert tuple(out.shape) == (3, 2)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        shape_hints=["indices:3,1"],
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        shape_hints=["indices:3,1"],
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_avoids_python_conditional_for_dynamic_reshape_shape_tensor(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_shape_tensor_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape_x"] = TensorIR(
        name="shape_x",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["axis0"] = TensorIR(
        name="axis0",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["axis1"] = TensorIR(
        name="axis1",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["zero"] = TensorIR(
        name="zero",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["dim0"] = TensorIR(
        name="dim0",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["dim1"] = TensorIR(
        name="dim1",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["dim0_is_zero"] = TensorIR(
        name="dim0_is_zero",
        dtype="BOOL",
        shape=[1],
        shape_signature=[1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["dim0_fixed"] = TensorIR(
        name="dim0_fixed",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="SHAPE", inputs=["x"], outputs=["shape_x"], options={}),
            OperatorIR(op_type="GATHER", inputs=["shape_x", "axis0"], outputs=["dim0"], options={"axis": 0, "batchDims": 0}),
            OperatorIR(op_type="GATHER", inputs=["shape_x", "axis1"], outputs=["dim1"], options={"axis": 0, "batchDims": 0}),
            OperatorIR(op_type="EQUAL", inputs=["dim0", "zero"], outputs=["dim0_is_zero"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="SELECT", inputs=["dim0_is_zero", "dim0", "dim0"], outputs=["dim0_fixed"], options={}),
            OperatorIR(op_type="CONCATENATION", inputs=["dim0_fixed", "dim1"], outputs=["reshape_shape"], options={"axis": 0, "fusedActivationFunction": "NONE"}),
            OperatorIR(op_type="RESHAPE", inputs=["x", "reshape_shape"], outputs=["y"], options={"allowZero": False}),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_shape_tensor_reshape_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_resolve_reshape_shape_tensor(reshape_shape, x, allow_zero=False)" in model_source
    assert "if (x.shape[0] == 0)" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (3, 1)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        shape_hints=["x:3,1"],
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        shape_hints=["x:3,1"],
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_supports_dynamic_gather_nd_params_for_torch_export(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_gather_nd_params_export")
    model_ir.inputs = ["params", "indices"]
    model_ir.outputs = ["y"]
    model_ir.tensors["params"] = TensorIR(
        name="params",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[-1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[-1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=["params", "indices"],
            outputs=["y"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_gather_nd_params_export_pkg"),
    )
    runtime_source = (Path(package_path) / "runtime.py").read_text(encoding="utf-8")
    assert "prefix_shape = list(indices_i64.shape[:-1])" in runtime_source
    assert "params[tuple(gather_indices)]" in runtime_source
    assert "prefix_shape = [int(dim) for dim in list(indices_i64.shape[:-1])]" not in runtime_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    params = torch.tensor(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[0], [2], [4]], dtype=torch.int32)
    out = cast(Any, model).forward_named(params=params, indices=indices)["y"]
    expected = params[indices[:, 0].to(dtype=torch.int64)]
    assert tuple(out.shape) == (3, 2)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        shape_hints=["params:5,2", "indices:3,1"],
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        shape_hints=["params:5,2", "indices:3,1"],
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_handles_same_max_pool_on_nhwc_inputs(tmp_path) -> None:
    model_ir = ModelIR(name="pool_nhwc_same_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 5, 1],
        shape_signature=[1, 4, 5, 1],
        logical_layout="NHWC",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 5, 1],
        shape_signature=[1, 4, 5, 1],
        logical_layout="NHWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="MAX_POOL_2D",
            inputs=["x"],
            outputs=["y"],
            options={
                "filterHeight": 3,
                "filterWidth": 3,
                "strideH": 1,
                "strideW": 1,
                "padding": "SAME",
                "fusedActivationFunction": "NONE",
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "pool_nhwc_same_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "def _max_pool2d_same" not in model_source
    assert "def _avg_pool2d_same" not in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = F.max_pool2d(
        F.pad(x, [1, 1, 1, 1], mode="constant", value=float("-inf")),
        kernel_size=(3, 3),
        stride=(1, 1),
    )
    assert torch.equal(out, expected)


def test_export_pytorch_package_handles_same_max_pool_stride2_odd_input(tmp_path) -> None:
    model_ir = ModelIR(name="pool_same_stride2_odd_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="MAX_POOL_2D",
            inputs=["x"],
            outputs=["y"],
            options={
                "filterHeight": 3,
                "filterWidth": 3,
                "strideH": 2,
                "strideW": 2,
                "padding": "SAME",
                "fusedActivationFunction": "NONE",
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "pool_same_stride2_odd_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = F.max_pool2d(
        F.pad(x, [1, 1, 1, 1], mode="constant", value=float("-inf")),
        kernel_size=(3, 3),
        stride=(2, 2),
    )
    assert out.shape == (1, 1, 4, 4)
    assert torch.equal(out, expected)


def test_export_pytorch_package_omits_unused_same_pool_helpers(tmp_path) -> None:
    model_ir = ModelIR(name="atan_without_pool_helpers")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NHWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ATAN",
            inputs=["x"],
            outputs=["y"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "atan_without_pool_helpers_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "def _max_pool2d_same" not in model_source
    assert "def _avg_pool2d_same" not in model_source
    assert "def _device" not in model_source
    assert "    _module_device,\n" not in model_source


def test_export_dynamo_onnx_handles_static_shape_input_reshape(tmp_path) -> None:
    model_ir = ModelIR(name="static_shape_input_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
        logical_layout="NHWC",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 16], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 16],
        shape_signature=[1, 16],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={"newShape": [1, 16]},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "static_shape_input_reshape_pkg"),
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()


def test_export_dynamo_onnx_and_exported_program_handle_constant_negative_one_reshape(tmp_path) -> None:
    model_ir = ModelIR(name="constant_negative_one_reshape")
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
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "constant_negative_one_reshape_pkg"),
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_scalarize_singleton_binary_rhs_constants(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_dynamic_binary_scalar_rhs_model_ir(),
        output_folder_path=str(tmp_path / "dynamic_binary_scalar_rhs_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_align_binary_inputs(x, torch.as_tensor([0]" not in model_source
    assert "_align_binary_inputs(x, torch.as_tensor([19248]" not in model_source
    assert "torch.lt(x, 0)" in model_source
    assert "torch.add(x, 19248)" in model_source

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_handle_constant_transpose_perm_tensor(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_unknown_layout_transpose_model_ir(),
        output_folder_path=str(tmp_path / "constant_transpose_perm_pkg"),
    )
    torchscript_path = export_torchscript_from_generated_package(
        package_dir=package_path,
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_handle_non_max_suppression_v4(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_non_max_suppression_v4_model_ir(),
        output_folder_path=str(tmp_path / "non_max_suppression_v4_pkg"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [10.0, 10.0, 11.0, 11.0], [20.0, 20.0, 21.0, 21.0], [30.0, 30.0, 31.0, 31.0]]],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.95, 0.90, 0.70, 0.20, 0.80]]], dtype=torch.float32)
    outputs = cast(Any, model).forward_named(boxes=boxes, scores=scores)
    assert outputs["selected_indices"].tolist() == [0, 4, 2]
    assert int(outputs["valid_count"].item()) == 3

    torchscript_path = export_torchscript_from_generated_package(
        package_dir=package_path,
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_handle_shape_derived_non_max_suppression_v4(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_shape_derived_non_max_suppression_v4_model_ir(),
        output_folder_path=str(tmp_path / "shape_derived_non_max_suppression_v4_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "def _run_nms_0(self, boxes: torch.Tensor, scores: torch.Tensor)" in model_source
    assert "max_output_size: torch.Tensor" not in model_source
    assert "torch.as_tensor(5, dtype=torch.int32" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [10.0, 10.0, 11.0, 11.0], [20.0, 20.0, 21.0, 21.0], [30.0, 30.0, 31.0, 31.0]],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.95, 0.90, 0.70, 0.20, 0.80], dtype=torch.float32)
    outputs = cast(Any, model).forward_named(boxes=boxes, scores=scores)
    assert outputs["selected_indices"].tolist() == [0, 4, 2, 0, 0]
    assert int(outputs["valid_count"].item()) == 3

    torchscript_path = export_torchscript_from_generated_package(
        package_dir=package_path,
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_skip_shape_specialized_non_max_suppression_v4(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_dynamic_shape_derived_non_max_suppression_v4_model_ir(),
        output_folder_path=str(tmp_path / "dynamic_shape_derived_non_max_suppression_v4_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "max_output_size: torch.Tensor" not in model_source

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        raise_on_failure=False,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        raise_on_failure=False,
    )
    assert dynamo_onnx_path is None
    assert exported_program_path is None

    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert "data-dependent NON_MAX_SUPPRESSION_V4 parameters" in metadata["dynamo_onnx"]["skipped_reason"]
    assert "data-dependent NON_MAX_SUPPRESSION_V4 parameters" in metadata["exported_program"]["skipped_reason"]


def test_export_torchscript_handles_data_dependent_non_max_suppression_v4(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_dynamic_shape_derived_non_max_suppression_v4_model_ir(),
        output_folder_path=str(tmp_path / "dynamic_shape_derived_non_max_suppression_v4_torchscript_pkg"),
    )
    torchscript_path = export_torchscript_from_generated_package(
        package_dir=package_path,
        raise_on_failure=False,
    )
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()

    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["torchscript"]["file_name"] is not None
    assert metadata["torchscript"]["trace_mode"] in {"trace", "script"}


def test_export_artifacts_skip_data_dependent_non_max_suppression_post_processing(tmp_path) -> None:
    package_dir = tmp_path / "data_dependent_non_max_suppression_post_processing_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (package_dir / "model.py").write_text(
        "\n".join(
            [
                "import torch",
                "",
                "class Model(torch.nn.Module):",
                "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
                "        selected_indices_nms_valid_count_scalar_c0 = x",
                "        selected_indices_nms_valid_indices_c0 = torch.arange(",
                "            start=0,",
                "            end=selected_indices_nms_valid_count_scalar_c0.reshape(-1)[0].item(),",
                "            step=1,",
                "            device=selected_indices_nms_valid_count_scalar_c0.device,",
                "            dtype=selected_indices_nms_valid_count_scalar_c0.dtype,",
                "        )",
                "        return selected_indices_nms_valid_indices_c0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=str(package_dir),
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=str(package_dir),
    )
    assert dynamo_onnx_path is None
    assert exported_program_path is None

    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "skipped_reason" in metadata["dynamo_onnx"]
    assert "NON_MAX_SUPPRESSION" in metadata["dynamo_onnx"]["skipped_reason"]
    assert "skipped_reason" in metadata["exported_program"]
    assert "NON_MAX_SUPPRESSION" in metadata["exported_program"]["skipped_reason"]
    assert "error" not in metadata["dynamo_onnx"]
    assert "error" not in metadata["exported_program"]


def test_export_artifacts_skip_non_native_saved_model_backed_package(tmp_path) -> None:
    package_dir = tmp_path / "saved_model_backed_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "model.py").write_text("import torch\n", encoding="utf-8")
    (package_dir / "metadata.json").write_text(
        json.dumps(
            {
                "execution_backend": "saved_model",
                "inputs": ["x"],
                "outputs": ["y"],
                "tensors": {
                    "x": {"shape": [1], "shape_signature": [1]},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    torchscript_path = export_torchscript_from_generated_package(
        package_dir=str(package_dir),
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=str(package_dir),
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=str(package_dir),
    )
    assert torchscript_path is None
    assert dynamo_onnx_path is None
    assert exported_program_path is None

    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "skipped_reason" in metadata["torchscript"]
    assert "execution_backend=saved_model" in metadata["torchscript"]["skipped_reason"]
    assert "skipped_reason" in metadata["dynamo_onnx"]
    assert "execution_backend=saved_model" in metadata["dynamo_onnx"]["skipped_reason"]
    assert "skipped_reason" in metadata["exported_program"]
    assert "execution_backend=saved_model" in metadata["exported_program"]["skipped_reason"]
    assert "error" not in metadata["torchscript"]
    assert "error" not in metadata["dynamo_onnx"]
    assert "error" not in metadata["exported_program"]


def test_export_artifacts_handle_runtime_minus_one_shape_tensor(tmp_path) -> None:
    model_ir = ModelIR(name="runtime_minus_one_shape_tensor")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([-1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
    )
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={"newShape": []})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "runtime_minus_one_shape_tensor_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_resolve_reshape_shape([-1, 1], x, allow_zero=False)" in model_source
    assert "_resolve_reshape_shape(shape," not in model_source

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_handle_shape_tensor_driven_reshape(tmp_path) -> None:
    model_ir = ModelIR(name="shape_tensor_driven_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["one"] = TensorIR(
        name="one",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["x_shape"] = TensorIR(
        name="x_shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="SHAPE",
                inputs=["x"],
                outputs=["x_shape"],
                options={"outType": "INT32"},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["x_shape", "one"],
                outputs=["reshape_shape"],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["x", "reshape_shape"],
                outputs=["y"],
                options={},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "shape_tensor_driven_reshape_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.tensor(list(" not in model_source
    assert "_shape_tensor(x" in model_source
    assert "reshape_shape.to(dtype=torch.int64).reshape(-1)" in model_source
    assert "_resolve_reshape_shape_tensor(" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(5, dtype=torch.float32)
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (5, 1)
    assert torch.equal(out, x.reshape(5, 1))

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_preserves_runtime_zero_dims_in_shape_tensor_reshape(tmp_path) -> None:
    model_ir = ModelIR(name="shape_tensor_zero_dim_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[-1, 4],
    )
    model_ir.tensors["one"] = TensorIR(
        name="one",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["x_shape"] = TensorIR(
        name="x_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 4],
        shape_signature=[1, -1, 4],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="SHAPE",
                inputs=["x"],
                outputs=["x_shape"],
                options={"outType": "INT32"},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["one", "x_shape"],
                outputs=["reshape_shape"],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["x", "reshape_shape"],
                outputs=["y"],
                options={},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "shape_tensor_zero_dim_reshape_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "reshape_shape.to(dtype=torch.int64).reshape(-1)" in model_source
    assert "_resolve_reshape_shape_tensor(" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.empty((0, 4), dtype=torch.float32)
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (1, 0, 4)
    assert out.numel() == 0


def test_export_artifacts_handle_dynamic_fill_shape_tensor(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_fill_shape_tensor")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["value"] = TensorIR(
        name="value",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1.5, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="SHAPE",
                inputs=["x"],
                outputs=["shape"],
                options={"outType": "INT32"},
            ),
            OperatorIR(
                op_type="FILL",
                inputs=["shape", "value"],
                outputs=["y"],
                options={},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_fill_shape_tensor_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_shape_tensor(x, dtype=torch.int32, device=x.device)" in model_source
    assert "torch.full(_shape_list(x), 1.5" in model_source
    assert "dtype=torch.float32, device=x.device" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(5, dtype=torch.float32)
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (5,)
    assert torch.equal(out, torch.full((5,), 1.5, dtype=torch.float32))

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_handle_dynamic_tile_without_aten_tile(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_tile_without_aten_tile")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 256, 1, 1],
        shape_signature=[1, 256, -1, 1],
    )
    model_ir.tensors["multiples"] = TensorIR(
        name="multiples",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 7, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 256, 7, 1],
        shape_signature=[1, 256, -1, 1],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="TILE",
            inputs=["x", "multiples"],
            outputs=["y"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_tile_without_aten_tile_pkg"),
    )
    runtime_source = (Path(package_path) / "runtime.py").read_text(encoding="utf-8")
    assert "def _apply_tile_eager" in runtime_source
    assert "aten.tile.default" not in runtime_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 256 * 5 * 1, dtype=torch.float32).reshape(1, 256, 5, 1)
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (1, 256, 35, 1)
    assert torch.equal(out, torch.tile(x, (1, 1, 7, 1)))

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
        shape_hints=["x:1,256,5,1"],
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
        shape_hints=["x:1,256,5,1"],
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_artifacts_handle_dynamic_strided_slice_max_stop_literal(tmp_path) -> None:
    model_ir = ModelIR(name="dynamic_strided_slice_max_stop_literal")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
    )
    model_ir.tensors["begin"] = TensorIR(
        name="begin",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([0, 0], dtype=np.int32),
    )
    model_ir.tensors["end"] = TensorIR(
        name="end",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2147483647, 1], dtype=np.int32),
    )
    model_ir.tensors["strides"] = TensorIR(
        name="strides",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="STRIDED_SLICE",
            inputs=["x", "begin", "end", "strides"],
            outputs=["y"],
            options={"beginMask": 0, "endMask": 0},
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "dynamic_strided_slice_max_stop_literal_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "2147483647" not in model_source
    assert "[:, :1]" in model_source or "[0:, 0:1]" in model_source or "[:,:1]" in model_source

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    exported_program_path = export_exported_program_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_conv2d_same_stride2_uses_explicit_same_upper_padding(tmp_path) -> None:
    model_ir = _make_conv2d_same_stride2_relu_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_same_stride2_relu_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "pad=[0, 1, 0, 1]" in model_source
    assert "padding=(0, 0)" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.from_numpy(np.arange(1 * 3 * 4 * 4, dtype=np.float32).reshape(1, 3, 4, 4) / 10.0)
    with torch.no_grad():
        out = model(x)

    w = model.conv_block_0.conv.weight.detach()
    b = model.conv_block_0.conv.bias.detach()
    expected = torch.relu(F.conv2d(F.pad(x, [0, 1, 0, 1]), w, b, stride=(2, 2), padding=(0, 0)))
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_pytorch_package_handles_constant_public_output_in_staged_native_codegen(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="constant_public_output_staged_native")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y", "labels"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
    )
    model_ir.tensors["labels"] = TensorIR(
        name="labels",
        dtype="INT64",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 2, 3, 4], dtype=np.int64),
    )

    previous_name = "x"
    for op_index in range(90):
        output_name = "y" if op_index == 89 else f"identity_{op_index}"
        model_ir.tensors[output_name] = TensorIR(
            name=output_name,
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[1, 4],
        )
        model_ir.operators.append(
            OperatorIR(
                op_type="IDENTITY",
                inputs=[previous_name],
                outputs=[output_name],
                options={},
            )
        )
        previous_name = output_name

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "constant_public_output_staged_native_pkg"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("execution_backend") == "native"

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    outputs = cast(Any, model).forward_named(x=x)
    assert torch.allclose(outputs["y"], x)
    assert outputs["labels"].tolist() == [1, 2, 3, 4]


def test_export_pytorch_package_fuses_pad_before_permuted_conv_without_padding_channel_dim(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="conv2d_pad_fusion_prepermute")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["pads"] = TensorIR(
        name="pads",
        dtype="INT32",
        shape=[4, 2],
        shape_signature=[4, 2],
        data=np.asarray([[0, 0], [0, 1], [0, 1], [0, 0]], dtype=np.int32),
    )
    model_ir.tensors["x_padded"] = TensorIR(
        name="x_padded",
        dtype="FLOAT32",
        shape=[1, 5, 5, 3],
        shape_signature=[1, 5, 5, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[8, 3, 3, 3],
        shape_signature=[8, 3, 3, 3],
        data=(np.arange(8 * 3 * 3 * 3, dtype=np.float32).reshape(8, 3, 3, 3) / 50.0),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 2, 8],
        shape_signature=[1, 2, 2, 8],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="PAD",
                inputs=["x", "pads"],
                outputs=["x_padded"],
                options={},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_padded", "w", "b"],
                outputs=["y"],
                options={
                    "strideH": 2,
                    "strideW": 2,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "VALID",
                    "fusedActivationFunction": "NONE",
                },
            ),
        ]
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_pad_fusion_prepermute_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "pad=[0, 1, 0, 1]" in model_source
    assert "pad=[0, 0, 0, 1, 0, 1]" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 4 * 4 * 3, dtype=torch.float32).reshape(1, 4, 4, 3) / 10.0
    with torch.no_grad():
        out = model(x)

    expected = model.conv_block_0(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_pytorch_package_cancels_boundary_transpose_before_padded_conv(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="conv2d_boundary_transpose_pad_fusion")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["pads"] = TensorIR(
        name="pads",
        dtype="INT32",
        shape=[4, 2],
        shape_signature=[4, 2],
        data=np.asarray([[0, 0], [0, 1], [0, 1], [0, 0]], dtype=np.int32),
    )
    model_ir.tensors["x_padded"] = TensorIR(
        name="x_padded",
        dtype="FLOAT32",
        shape=[1, 5, 5, 3],
        shape_signature=[1, 5, 5, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[5, 3, 3, 3],
        shape_signature=[5, 3, 3, 3],
        data=(np.arange(5 * 3 * 3 * 3, dtype=np.float32).reshape(5, 3, 3, 3) / 50.0),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[5],
        shape_signature=[5],
        data=np.linspace(-0.2, 0.2, 5, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 2, 5],
        shape_signature=[1, 2, 2, 5],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["x", "perm"],
                outputs=["x_nhwc"],
                options={},
            ),
            OperatorIR(
                op_type="PAD",
                inputs=["x_nhwc", "pads"],
                outputs=["x_padded"],
                options={},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_padded", "w", "b"],
                outputs=["y"],
                options={
                    "strideH": 2,
                    "strideW": 2,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "VALID",
                    "fusedActivationFunction": "NONE",
                },
            ),
        ]
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_boundary_transpose_pad_fusion_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.conv_block_0(x)" in model_source
    assert "self.conv_block_0(x.permute(0, 3, 1, 2).contiguous())" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4) / 10.0
    with torch.no_grad():
        out = model(x)

    expected = model.conv_block_0(x).permute(0, 2, 3, 1).contiguous()
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_cancels_implicit_boundary_transpose_before_padded_conv(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="conv2d_implicit_boundary_transpose_pad_fusion")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
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
        data=np.asarray([[0, 0], [0, 1], [0, 1], [0, 0]], dtype=np.int32),
    )
    model_ir.tensors["x_padded"] = TensorIR(
        name="x_padded",
        dtype="FLOAT32",
        shape=[1, 5, 5, 3],
        shape_signature=[1, 5, 5, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[5, 3, 3, 3],
        shape_signature=[5, 3, 3, 3],
        data=(np.arange(5 * 3 * 3 * 3, dtype=np.float32).reshape(5, 3, 3, 3) / 50.0),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[5],
        shape_signature=[5],
        data=np.linspace(-0.2, 0.2, 5, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 2, 5],
        shape_signature=[1, 2, 2, 5],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="PAD",
                inputs=["x", "pads"],
                outputs=["x_padded"],
                options={},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_padded", "w", "b"],
                outputs=["y"],
                options={
                    "strideH": 2,
                    "strideW": 2,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "VALID",
                    "fusedActivationFunction": "NONE",
                },
            ),
        ]
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_implicit_boundary_transpose_pad_fusion_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.conv_block_0(x)" in model_source
    assert "self.conv_block_0(x.permute(0, 3, 1, 2).contiguous())" not in model_source


def test_export_pytorch_package_permutes_ambiguous_nhwc_pointwise_conv_inputs(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="conv2d_ambiguous_nhwc_pointwise")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 5, 4],
        shape_signature=[1, 4, 5, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[6, 1, 1, 4],
        shape_signature=[6, 1, 1, 4],
        data=(np.arange(6 * 4, dtype=np.float32).reshape(6, 1, 1, 4) / 20.0),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[6],
        shape_signature=[6],
        data=np.linspace(-0.3, 0.3, 6, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 5, 6],
        shape_signature=[1, 4, 5, 6],
        logical_layout="NHWC",
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
                "padding": "VALID",
                "fusedActivationFunction": "NONE",
            },
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_ambiguous_nhwc_pointwise_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.conv_block_0(x.permute(0, 3, 1, 2).contiguous())" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 4 * 5 * 4, dtype=torch.float32).reshape(1, 4, 5, 4) / 10.0
    with torch.no_grad():
        out = model(x)

    expected = model.conv_block_0(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_prefers_logical_layout_for_ambiguous_nhwc_to_nchw_conv_inputs(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="conv2d_ambiguous_nhwc_to_nchw")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 48, 128, 48],
        shape_signature=[1, 48, 128, 48],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[24, 1, 1, 48],
        shape_signature=[24, 1, 1, 48],
        data=(np.arange(24 * 48, dtype=np.float32).reshape(24, 1, 1, 48) / 100.0),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[24],
        shape_signature=[24],
        data=np.linspace(-0.3, 0.3, 24, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 24, 48, 128],
        shape_signature=[1, 24, 48, 128],
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
                "padding": "VALID",
                "fusedActivationFunction": "NONE",
            },
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_ambiguous_nhwc_to_nchw_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.conv_block_0(x.permute(0, 3, 1, 2).contiguous())" in model_source
    assert "permute(0, 1, 3, 2)" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 48 * 128 * 48, dtype=torch.float32).reshape(1, 48, 128, 48) / 50.0
    with torch.no_grad():
        out = model(x)

    expected = model.conv_block_0(x.permute(0, 3, 1, 2).contiguous())
    assert tuple(out.shape) == (1, 24, 48, 128)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_export_generated_package_bridges_reshaped_nwc_input_into_channel_last_conv_block(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="reshaped_nwc_channel_last_conv_block")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 64, 64],
        shape_signature=[1, 64, 64],
        logical_layout="NWC",
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 64, 64, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 64, 64, 1],
        shape_signature=[1, 64, 64, 1],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[64, 7, 7, 1],
        shape_signature=[64, 7, 7, 1],
        data=(np.arange(64 * 7 * 7, dtype=np.float32).reshape(64, 7, 7, 1) / 500.0),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[64],
        shape_signature=[64],
        data=np.linspace(-0.2, 0.2, 64, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 64, 64, 64],
        shape_signature=[1, 64, 64, 64],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="RESHAPE",
                inputs=["x", "reshape_shape"],
                outputs=["x_nhwc"],
                options={"newShape": [1, 64, 64, 1]},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x_nhwc", "w", "b"],
                outputs=["y"],
                options={
                    "strideH": 1,
                    "strideW": 1,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "SAME",
                    "fusedActivationFunction": "NONE",
                },
            ),
        ]
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshaped_nwc_channel_last_conv_block_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".permute(0, 3, 1, 2).contiguous()" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 64 * 64, dtype=torch.float32).reshape(1, 64, 64) / 100.0
    with torch.no_grad():
        out = model(x)

    expected = (
        model.conv_block_0(x.reshape(1, 64, 64, 1).permute(0, 3, 1, 2).contiguous())
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_generated_package_elides_batchless_rank3_public_output_transpose(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="batchless_rank3_public_output_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["batchless_rank3_public_boundary_names"] = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 8, 8],
        shape_signature=[1, 8, 8],
        logical_layout="NWC",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 2, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 8, 8],
        shape_signature=[1, 8, 8],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["y"], options={"perm": [0, 2, 1]})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "batchless_rank3_public_output_transpose_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_torch_permute(x, [0, 2, 1])" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8)
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_export_generated_package_elides_stale_square_layout_transposes_around_channel_last_norm(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="stale_square_layout_transpose_norm")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["perm_cf"] = TensorIR(
        name="perm_cf",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["x_cf"] = TensorIR(
        name="x_cf",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 2], dtype=np.int32),
    )
    model_ir.tensors["mean"] = TensorIR(
        name="mean",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["centered"] = TensorIR(
        name="centered",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["scale"] = TensorIR(
        name="scale",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
        data=np.linspace(0.8, 1.2, 8, dtype=np.float32).reshape(1, 1, 1, 8),
    )
    model_ir.tensors["scaled"] = TensorIR(
        name="scaled",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_cl"] = TensorIR(
        name="perm_cl",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["scaled_nhwc_bridge"] = TensorIR(
        name="scaled_nhwc_bridge",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
        data=np.linspace(-0.2, 0.2, 8, dtype=np.float32).reshape(1, 1, 1, 8),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_cf"], outputs=["x_cf"], options={}),
            OperatorIR(
                op_type="MEAN",
                inputs=["x_cf", "axes"],
                outputs=["mean"],
                options={"keepDims": True},
            ),
            OperatorIR(
                op_type="SUB",
                inputs=["x_cf", "mean"],
                outputs=["centered"],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="MUL",
                inputs=["centered", "scale"],
                outputs=["scaled"],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="TRANSPOSE", inputs=["scaled", "perm_cl"], outputs=["scaled_nhwc_bridge"], options={}),
            OperatorIR(
                op_type="ADD",
                inputs=["scaled_nhwc_bridge", "bias"],
                outputs=["y"],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "stale_square_layout_transpose_norm_pkg"),
    )

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8) / 50.0
    with torch.no_grad():
        out = model(x)

    expected_mean = torch.mean(x, dim=(1, 2), keepdim=True)
    expected = ((x - expected_mean) * torch.from_numpy(model_ir.tensors["scale"].data)).permute(0, 2, 3, 1)
    expected = expected.contiguous() + torch.from_numpy(model_ir.tensors["bias"].data)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_generated_package_preserves_same_shape_layout_bridge_transpose_before_channel_last_bias_add(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="preserve_same_shape_layout_bridge_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm_cl"] = TensorIR(
        name="perm_cl",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
        data=np.linspace(-0.2, 0.2, 8, dtype=np.float32).reshape(1, 1, 1, 8),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_cl"], outputs=["x_nhwc"], options={}),
            OperatorIR(
                op_type="ADD",
                inputs=["x_nhwc", "bias"],
                outputs=["y"],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "preserve_same_shape_layout_bridge_transpose_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "x = _torch_permute(x, [0, 2, 3, 1])" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8) / 50.0
    with torch.no_grad():
        out = model(x)

    expected = x.permute(0, 2, 3, 1).contiguous() + torch.from_numpy(model_ir.tensors["bias"].data)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_generated_package_aligns_symint_shapes_for_channel_last_gap_conv_chain(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="channel_last_gap_conv_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 5, 4],
        shape_signature=[1, 4, 5, 4],
        logical_layout="NHWC",
    )
    model_ir.tensors["w0"] = TensorIR(
        name="w0",
        dtype="FLOAT32",
        shape=[8, 1, 1, 4],
        shape_signature=[8, 1, 1, 4],
        data=(np.arange(8 * 4, dtype=np.float32).reshape(8, 1, 1, 4) / 25.0),
    )
    model_ir.tensors["b0"] = TensorIR(
        name="b0",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.linspace(-0.2, 0.2, 8, dtype=np.float32),
    )
    model_ir.tensors["conv0"] = TensorIR(
        name="conv0",
        dtype="FLOAT32",
        shape=[1, 4, 5, 8],
        shape_signature=[1, 4, 5, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["gap_axes"] = TensorIR(
        name="gap_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 2], dtype=np.int32),
    )
    model_ir.tensors["gap"] = TensorIR(
        name="gap",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["w1"] = TensorIR(
        name="w1",
        dtype="FLOAT32",
        shape=[3, 1, 1, 8],
        shape_signature=[3, 1, 1, 8],
        data=(np.arange(3 * 8, dtype=np.float32).reshape(3, 1, 1, 8) / 15.0),
    )
    model_ir.tensors["b1"] = TensorIR(
        name="b1",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.linspace(-0.1, 0.1, 3, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 3],
        shape_signature=[1, 1, 1, 3],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x", "w0", "b0"],
                outputs=["conv0"],
                options={
                    "strideH": 1,
                    "strideW": 1,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "VALID",
                    "fusedActivationFunction": "NONE",
                },
            ),
            OperatorIR(
                op_type="MEAN",
                inputs=["conv0", "gap_axes"],
                outputs=["gap"],
                options={"keepDims": True},
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["gap", "w1", "b1"],
                outputs=["y"],
                options={
                    "strideH": 1,
                    "strideW": 1,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "VALID",
                    "fusedActivationFunction": "NONE",
                },
            ),
        ]
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "channel_last_gap_conv_chain_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.conv_block_0(x.permute(0, 3, 1, 2).contiguous())" in model_source
    assert "self.conv_block_1(x.permute(0, 3, 1, 2).contiguous())" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 4 * 5 * 4, dtype=torch.float32).reshape(1, 4, 5, 4) / 10.0
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (1, 1, 1, 3)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_generated_package_permutes_ambiguous_channel_last_conv_outputs_before_gap(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="ambiguous_channel_last_conv_gap")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 8],
        shape_signature=[8, 1, 1, 8],
        data=(np.arange(8 * 8, dtype=np.float32).reshape(8, 1, 1, 8) / 17.0),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.linspace(-0.4, 0.4, 8, dtype=np.float32),
    )
    model_ir.tensors["conv"] = TensorIR(
        name="conv",
        dtype="FLOAT32",
        shape=[1, 8, 8, 8],
        shape_signature=[1, 8, 8, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
        logical_layout="NHWC",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x", "w", "b"],
                outputs=["conv"],
                options={
                    "strideH": 1,
                    "strideW": 1,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "padding": "VALID",
                    "fusedActivationFunction": "NONE",
                },
            ),
            OperatorIR(
                op_type="MEAN",
                inputs=["conv", "axes"],
                outputs=["y"],
                options={"keepDims": True},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "ambiguous_channel_last_conv_gap_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".permute(0, 2, 3, 1).contiguous()" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8) / 13.0
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = model.conv_block_0(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    expected = torch.mean(expected, dim=(1, 2), keepdim=True)
    assert tuple(out.shape) == (1, 1, 1, 8)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_export_artifacts_handle_atan2_native_codegen(tmp_path) -> None:
    model_ir = ModelIR(name="atan2_native_codegen")
    model_ir.inputs = ["y", "x"]
    model_ir.outputs = ["z"]
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ATAN2",
            inputs=["y", "x"],
            outputs=["z"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "atan2_native_codegen_pkg"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "native"
    model_source = (package_dir / "model.py").read_text(encoding="utf-8")
    assert "torch.atan2(y, x)" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    y = torch.linspace(-0.8, 0.8, steps=48, dtype=torch.float32).reshape(1, 3, 4, 4)
    x = torch.linspace(0.2, 1.6, steps=48, dtype=torch.float32).reshape(1, 3, 4, 4)
    out = cast(Any, model).forward_named(y=y, x=x)["z"]
    assert torch.allclose(out, torch.atan2(y, x), atol=1e-6, rtol=1e-6)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_generated_package_elides_trailing_axis_reshape_for_1d_constant_binary_rhs(tmp_path) -> None:
    model_ir = ModelIR(name="trailing_axis_1d_constant_binary_rhs")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 5, 8],
        shape_signature=[-1, 5, 8],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["scale"] = TensorIR(
        name="scale",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.linspace(0.5, 1.2, 8, dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 8],
        shape_signature=[-1, 5, 8],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(op_type="MUL", inputs=["x", "scale"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "trailing_axis_1d_constant_binary_rhs_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".reshape([1, 1, 8])" not in model_source
    assert "torch.mul(x, torch.as_tensor(" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = x * torch.linspace(0.5, 1.2, 8, dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_generated_package_uses_broadcast_buffer_alias_for_large_trailing_axis_constant_binary_rhs(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="buffered_trailing_axis_1d_constant_binary_rhs")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 5, 64],
        shape_signature=[-1, 5, 64],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["scale"] = TensorIR(
        name="scale",
        dtype="FLOAT32",
        shape=[64],
        shape_signature=[64],
        data=np.linspace(0.5, 1.2, 64, dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 64],
        shape_signature=[-1, 5, 64],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(op_type="MUL", inputs=["x", "scale"], outputs=["y"], options={})
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "buffered_trailing_axis_1d_constant_binary_rhs_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".reshape([1, 1, 64])" not in model_source
    assert "register_buffer('const_scale_broadcast_1x1x64'" in model_source
    assert "torch.mul(x, self.const_scale_broadcast_1x1x64)" in model_source

    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    constant_reshape_inputs = {
        getattr(node.args[0], "name", "")
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.reshape.default"
        and "const_scale" in getattr(node.args[0], "name", "")
    }
    assert constant_reshape_inputs == set()

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(320, dtype=torch.float32).reshape(1, 5, 64)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = x * torch.linspace(0.5, 1.2, 64, dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_generated_package_aligns_transposed_gather_nd_params(tmp_path) -> None:
    model_ir = ModelIR(name="transpose_gather_nd_params")
    model_ir.inputs = ["data", "indices"]
    model_ir.outputs = ["output"]
    model_ir.tensors["data"] = TensorIR(
        name="data",
        dtype="FLOAT32",
        shape=[1, 17, 48, 64],
        shape_signature=[1, 17, 48, 64],
        logical_layout="NCHW",
    )
    model_ir.tensors["data_transpose_perm"] = TensorIR(
        name="data_transpose_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["data_nhwc"] = TensorIR(
        name="data_nhwc",
        dtype="FLOAT32",
        shape=[1, 48, 64, 17],
        shape_signature=[1, 48, 64, 17],
        logical_layout="NHWC",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT64",
        shape=[6, 3],
        shape_signature=[6, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[6, 17],
        shape_signature=[6, 17],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["data", "data_transpose_perm"],
                outputs=["data_nhwc"],
                options={},
            ),
            OperatorIR(
                op_type="GATHER_ND",
                inputs=["data_nhwc", "indices"],
                outputs=["output"],
                options={},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "transpose_gather_nd_params_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_torch_permute(data, [0, 2, 3, 1])" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    data = torch.arange(1 * 17 * 48 * 64, dtype=torch.float32).reshape(1, 17, 48, 64) / 100.0
    indices = torch.tensor(
        [[0, 0, 0], [0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 7, 8], [0, 9, 10]],
        dtype=torch.int64,
    )
    out = cast(Any, model).forward_named(data=data, indices=indices)["output"]
    expected_params = data.permute(0, 2, 3, 1).contiguous()
    expected = expected_params[tuple(indices[..., i] for i in range(indices.shape[-1]))]
    assert tuple(out.shape) == (6, 17)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_generated_package_folds_suffix_flatten_after_multi_index_gather(tmp_path) -> None:
    model_ir = ModelIR(name="gather_suffix_flatten_fold")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 10, 2],
        shape_signature=[1, 10, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[3, 2],
        shape_signature=[3, 2],
        data=np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["gathered"] = TensorIR(
        name="gathered",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 3, 4], dtype=np.int64),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4],
        shape_signature=[1, 3, 4],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="GATHER",
                inputs=["x", "indices"],
                outputs=["gathered"],
                options={"axis": 1, "batchDims": 0},
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["gathered", "shape"],
                outputs=["y"],
                options={"allowZero": False, "newShape": [1, 3, 4], "onnxRawNewShape": [1, 3, 4]},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "gather_suffix_flatten_fold_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.reshape(torch.index_select(x, 1," in model_source

    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert exported_program_path is not None
    exported_program = torch.export.load(str(exported_program_path))
    reshape_node_names = {
        node.name
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.reshape.default"
    }
    assert all(
        getattr(node.args[0], "name", "") not in reshape_node_names
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.reshape.default"
    )

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = torch.reshape(
        torch.index_select(
            x,
            1,
            torch.as_tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
        ),
        [1, 3, 4],
    )
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_merge_reference_public_boundary_metadata_bridges_imported_gather_nd_input(tmp_path) -> None:
    reference_model_ir = ModelIR(name="reference_transpose_gather_nd")
    reference_model_ir.inputs = ["data", "indices"]
    reference_model_ir.outputs = ["output"]
    reference_model_ir.tensors["data"] = TensorIR(
        name="data",
        dtype="FLOAT32",
        shape=[1, 17, 48, 64],
        shape_signature=[1, 17, 48, 64],
        logical_layout="NCHW",
    )
    reference_model_ir.tensors["data_transpose_perm"] = TensorIR(
        name="data_transpose_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    reference_model_ir.tensors["data_nhwc"] = TensorIR(
        name="data_nhwc",
        dtype="FLOAT32",
        shape=[1, 48, 64, 17],
        shape_signature=[1, 48, 64, 17],
        logical_layout="NHWC",
    )
    reference_model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[6, 3],
        shape_signature=[6, 3],
        logical_layout="UNKNOWN",
    )
    reference_model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[6, 17],
        shape_signature=[6, 17],
        logical_layout="UNKNOWN",
    )
    reference_model_ir.operators.extend(
        [
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["data", "data_transpose_perm"],
                outputs=["data_nhwc"],
                options={},
            ),
            OperatorIR(
                op_type="GATHER_ND",
                inputs=["data_nhwc", "indices"],
                outputs=["output"],
                options={},
            ),
        ]
    )

    imported_model_ir = ModelIR(name="imported_gather_nd")
    imported_model_ir.inputs = ["data", "indices"]
    imported_model_ir.outputs = ["output"]
    imported_model_ir.tensors["data"] = TensorIR(
        name="data",
        dtype="FLOAT32",
        shape=[1, 48, 64, 17],
        shape_signature=[1, 48, 64, 17],
        logical_layout="NHWC",
    )
    imported_model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[6, 3],
        shape_signature=[6, 3],
        logical_layout="UNKNOWN",
    )
    imported_model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[6, 17],
        shape_signature=[6, 17],
        logical_layout="UNKNOWN",
    )
    imported_model_ir.operators.append(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=["data", "indices"],
            outputs=["output"],
            options={},
        )
    )

    _merge_reference_public_boundary_metadata(
        imported_model_ir=imported_model_ir,
        reference_model_ir=reference_model_ir,
    )

    assert imported_model_ir.operators[0].op_type == "TRANSPOSE"
    assert imported_model_ir.tensors["data"].shape_signature == [1, 17, 48, 64]
    assert imported_model_ir.tensors["data"].logical_layout == "NCHW"

    package_path = export_pytorch_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=str(tmp_path / "imported_gather_nd_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_torch_permute(data, [0, 2, 3, 1])" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    data = torch.arange(1 * 17 * 48 * 64, dtype=torch.float32).reshape(1, 17, 48, 64) / 100.0
    indices = torch.tensor(
        [[0, 0, 0], [0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 7, 8], [0, 9, 10]],
        dtype=torch.int32,
    )
    out = cast(Any, model).forward_named(data=data, indices=indices)["output"]
    expected = data.permute(0, 2, 3, 1).contiguous()[tuple(indices[..., i] for i in range(indices.shape[-1]))]
    assert tuple(out.shape) == (6, 17)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_export_generated_package_bridges_boundary_metadata_gather_nd_input(tmp_path) -> None:
    model_ir = ModelIR(name="boundary_metadata_gather_nd")
    model_ir.inputs = ["data", "indices"]
    model_ir.outputs = ["output"]
    model_ir.tensors["data"] = TensorIR(
        name="data",
        dtype="FLOAT32",
        shape=[1, 48, 64, 17],
        shape_signature=[1, 48, 64, 17],
        logical_layout="NHWC",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[6, 3],
        shape_signature=[6, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[6, 17],
        shape_signature=[6, 17],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=["data", "indices"],
            outputs=["output"],
            options={},
        )
    )
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "data": [1, 17, 48, 64],
        "indices": [6, 3],
        "output": [6, 17],
    }
    model_ir.metadata["onnx_public_layout_map"] = {
        "data": "NCHW",
    }

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "boundary_metadata_gather_nd_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert "_torch_permute(data, [0, 2, 3, 1])" in model_source
    assert metadata["tensors"]["data"]["shape_signature"] == [1, 17, 48, 64]
    assert metadata["tensors"]["data"]["logical_layout"] == "NCHW"

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    data = torch.arange(1 * 17 * 48 * 64, dtype=torch.float32).reshape(1, 17, 48, 64) / 100.0
    indices = torch.tensor(
        [[0, 0, 0], [0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 7, 8], [0, 9, 10]],
        dtype=torch.int32,
    )
    out = cast(Any, model).forward_named(data=data, indices=indices)["output"]
    expected = data.permute(0, 2, 3, 1).contiguous()[tuple(indices[..., i] for i in range(indices.shape[-1]))]
    assert tuple(out.shape) == (6, 17)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_convert_flatbuffer_direct_face_liveness_outputs_pytorch_accuracy(tmp_path) -> None:
    model_path = Path(__file__).resolve().parents[1] / "face_liveness.onnx"
    if not model_path.exists():
        pytest.skip("face_liveness.onnx is not available in the repository")

    output_dir = tmp_path / "face_liveness_out"
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
    package_path = output_dir / "face_liveness_pytorch"
    report = evaluate_pytorch_package_outputs(
        onnx_graph=onnx.load(str(model_path)),
        package_dir=str(package_path),
        output_report_path=str(output_dir / "face_liveness_pytorch_accuracy_report.json"),
        num_samples=1,
    )
    assert report["evaluation_pass"] is True


def test_export_generated_package_preserves_rank4_channel_last_mlp_bridge_artifacts(tmp_path) -> None:
    onnx_model = _make_nhwc_mlp_bridge_model()
    model_ir = lower_onnx_to_ir(
        onnx_graph=onnx_model,
        output_file_name="nhwc_mlp_bridge",
        show_progress=False,
    )
    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)
    assert normalize_logical_layout(normalized.tensors["ln"].logical_layout) == "NHWC"
    assert normalize_logical_layout(normalized.tensors["fc1_bias"].logical_layout) == "NHWC"

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "nhwc_mlp_bridge_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_normalize_axes(" in model_source
    assert "torch.matmul(" in model_source
    assert "[1, 8, 8, 6]" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8) / 20.0
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (1, 8, 8, 4)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_generated_package_permutes_rank4_channel_last_depthwise_conv_inputs(tmp_path) -> None:
    onnx_model = _make_nhwc_depthwise_bridge_model()
    model_ir = lower_onnx_to_ir(
        onnx_graph=onnx_model,
        output_file_name="nhwc_depthwise_bridge",
        show_progress=False,
    )
    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)
    assert normalize_logical_layout(normalized.tensors["ln_dw"].logical_layout) == "NHWC"

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "nhwc_depthwise_bridge_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.conv_block_1(" in model_source
    assert ".permute(0, 3, 1, 2).contiguous()" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8) / 10.0
    out = cast(Any, model).forward_named(x=x)["y"]
    assert tuple(out.shape) == (1, 8, 8, 6)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_dynamo_onnx_handles_static_resize_size_literal(tmp_path) -> None:
    model_ir = ModelIR(name="static_resize_size_literal")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 8, 8],
        shape_signature=[1, 3, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["size"] = TensorIR(
        name="size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([16, 16], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 16, 16],
        shape_signature=[1, 3, 16, 16],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESIZE_BILINEAR",
            inputs=["x", "size"],
            outputs=["y"],
            options={
                "alignCorners": False,
                "halfPixelCenters": True,
            },
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "static_resize_size_literal_pkg"),
    )
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
        package_dir=package_path,
    )
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()


def test_convert_flatbuffer_direct_small_conv_pool_model_outputs_pytorch_artifacts_and_accuracy(tmp_path) -> None:
    model_path = tmp_path / "small_face_liveness_style.onnx"
    onnx.save(_make_small_face_liveness_style_model(), str(model_path))
    output_dir = tmp_path / "small_face_liveness_style_out"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        flatbuffer_direct_output_torchscript=True,
        flatbuffer_direct_output_dynamo_onnx=True,
        flatbuffer_direct_output_exported_program=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "small_face_liveness_style_pytorch"
    assert (package_path / "small_face_liveness_style_jit.pt").exists()
    assert (package_path / "small_face_liveness_style_dynamo.onnx").exists()
    exported_program_path = package_path / "small_face_liveness_style_ep.pt2"
    assert exported_program_path.exists()
    with zipfile.ZipFile(str(exported_program_path), "r") as archive:
        model_json_name = next(name for name in archive.namelist() if name.endswith("models/model.json"))
        model_json_payload = archive.read(model_json_name)
    assert b"\"stack_trace\"" not in model_json_payload
    report = evaluate_pytorch_package_outputs(
        onnx_graph=onnx.load(str(model_path)),
        package_dir=str(package_path),
        output_report_path=str(output_dir / "small_face_liveness_style_pytorch_accuracy_report.json"),
        num_samples=1,
    )
    assert report["evaluation_pass"] is True


def test_convert_flatbuffer_direct_finder_reduces_native_torch_layout_noise_when_model_is_available(tmp_path) -> None:
    model_path = Path("finder.onnx")
    if not model_path.exists():
        pytest.skip("finder.onnx is not available")

    output_dir = tmp_path / "finder_out"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        flatbuffer_direct_output_torchscript=True,
        flatbuffer_direct_output_dynamo_onnx=True,
        flatbuffer_direct_output_exported_program=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "finder_pytorch"
    metadata = json.loads((package_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "native"
    assert (package_path / "finder_jit.pt").exists()
    assert (package_path / "finder_dynamo.onnx").exists()
    model_source = (package_path / "model.py").read_text(encoding="utf-8")
    assert "image_public_layout_bridge =" not in model_source
    assert "self.conv_block_0(image)" in model_source
    assert "Conv_output_nhwc =" not in model_source
    assert "self.conv_block_12(" in model_source
    nhwc_dynamic = sorted(
        name
        for name, tensor in metadata["tensors"].items()
        if name not in set(metadata["inputs"]) | set(metadata["outputs"])
        and str(tensor.get("logical_layout", "")) == "NHWC"
        and len(list(tensor.get("shape", []))) == 4
        and not bool(tensor.get("has_data", False))
    )
    assert "image_public_layout_bridge" in nhwc_dynamic
    residual_internal_nhwc = {
        name for name in nhwc_dynamic if name != "image_public_layout_bridge"
    }
    assert residual_internal_nhwc == {
        next(name for name in residual_internal_nhwc if name.endswith("/backbone/model2/model2.5/Conv_output_nhwc"))
    }

    jit_module = torch.jit.load(str(package_path / "finder_jit.pt"))
    node_counts = Counter(node.kind() for node in jit_module.inlined_graph.nodes())
    assert node_counts.get("aten::permute", 0) <= 2
    assert node_counts.get("aten::contiguous", 0) <= 2

    pytorch_report = evaluate_pytorch_package_outputs(
        onnx_graph=onnx.load(str(model_path)),
        package_dir=str(package_path),
        output_report_path=str(output_dir / "finder_pytorch_accuracy_report.json"),
        num_samples=1,
    )
    assert pytorch_report["evaluation_pass"] is True


def test_convert_flatbuffer_direct_birdnet_preserves_permute_optimizations_and_pytorch_parity_when_model_is_available(
    tmp_path,
) -> None:
    model_path = Path("birdnet.onnx")
    if not model_path.exists():
        pytest.skip("birdnet.onnx is not available")

    output_dir = tmp_path / "birdnet_out"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        flatbuffer_direct_output_torchscript=True,
        flatbuffer_direct_output_dynamo_onnx=True,
        flatbuffer_direct_output_exported_program=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "birdnet_pytorch"
    metadata = json.loads((package_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "native"
    assert "error" not in metadata["torchscript"]
    assert "error" not in metadata["dynamo_onnx"]
    assert "error" not in metadata["exported_program"]
    assert (package_path / cast(str, metadata["torchscript"]["file_name"])).exists()
    assert (package_path / cast(str, metadata["dynamo_onnx"]["file_name"])).exists()
    assert (package_path / cast(str, metadata["exported_program"]["file_name"])).exists()

    model_source = (package_path / "model.py").read_text(encoding="utf-8")
    assigned_names = re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", model_source, flags=re.MULTILINE)
    assert "permute(0, 1, 3, 2)" not in model_source
    assert ".reshape(-1).tolist()" not in model_source
    assert "torch.flip(" in model_source
    assert max(len(name) for name in assigned_names) <= 40
    assert "input_nhwc__channel_first" not in model_source
    assert "output_nhwc__channel_first" not in model_source
    assert "__channel_first" not in model_source
    assert "_cf" in model_source
    assert any(re.search(r"_cf(?:_[0-9a-f]{4})?$", name) for name in assigned_names)
    assert re.search(r"\w+ = self\.conv_block_4\(", model_source) is not None
    assert re.search(r"self\.conv_block_5\(\w+_cf\)", model_source) is not None
    assert "self.conv_block_5(node_Conv_4_output_nhwc.permute(" not in model_source
    assert re.search(r"torch\.add\(\w+_cf, \w+_cf\)", model_source) is not None
    assert re.search(r"self\.conv_block_8\(\w+_cf\)", model_source) is not None
    assert "self.const_model_MEL_SPEC1_stft_hann_window_sub_2.reshape([1, 1, 2048])" not in model_source
    assert "self.const_model_MEL_SPEC2_stft_hann_window_sub_2.reshape([1, 1, 1024])" not in model_source
    assert "Transpose__1363_transposed_0 = " not in model_source
    assert re.search(r"torch\.mean\(\w+_cf, dim=\[2, 3\], keepdim=False\)", model_source) is not None
    assert re.search(r"\w+_cf = torch\.reshape\(\w+, \[1, 288, 1, 1\]\)", model_source) is not None
    assert re.search(r"self\.conv_block_13\(\w+_cf\)", model_source) is not None
    assert re.search(r"\w+_cf = self\.conv_block_14\(\w+_cf\)", model_source) is not None
    assert re.search(r"\w+_cf = torch\.mul\(\w+_cf, \w+_cf\)", model_source) is not None
    assert "_align_binary_inputs(Transpose__1363_transposed_0, model_BLOCK_2_1_SE_CONV_2_Sigmoid_Y_0" not in model_source
    assert re.search(r"\w+_cf = self\.conv_block_15\(\w+_cf\)", model_source) is not None
    assert "model_BLOCK_4_4_ADD_add_C_0__raw = _align_tensor_to_target_shape(model_BLOCK_4_4_ADD_add_C_0__raw__channel_first.permute(0, 2, 3, 1).contiguous(), [-1, 3, 8, 192])" not in model_source
    assert "_align_binary_inputs(model_BLOCK_4_4_ADD_add_C_0__raw, self.const_model_BNORM_POST_NOQUANT_FusedBatchNormV31" not in model_source
    assert "_channel_first_" not in model_source
    assert "const_model_BNORM_POST_NOQUANT_FusedBatchNormV31_channel_first_1x192x1x1" not in model_source
    assert "const_const_fold_opt__1698_channel_first_1x192x1x1" not in model_source
    assert re.search(r"self\.register_buffer\('const_[^']{1,40}', .*persistent=False\)", model_source) is not None
    assert (
        re.search(
            r"\w+_cf(?:_[0-9a-f]{4})? = torch\.mul\(\w+_cf(?:_[0-9a-f]{4})?, self\.const_[A-Za-z0-9_]+\)",
            model_source,
        )
        is not None
    )
    assert (
        re.search(
            r"\w+_cf(?:_[0-9a-f]{4})? = torch\.add\(\w+_cf(?:_[0-9a-f]{4})?, self\.const_[A-Za-z0-9_]+\)",
            model_source,
        )
        is not None
    )
    assert (
        re.search(
            r"\w+_cf(?:_[0-9a-f]{4})? = self\.conv_block_76\(\w+_cf(?:_[0-9a-f]{4})?\)",
            model_source,
        )
        is not None
    )

    exported_program = torch.export.load(str(package_path / cast(str, metadata["exported_program"]["file_name"])))
    constant_permute_inputs = {
        getattr(node.args[0], "name", "")
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.permute.default"
        and getattr(node.args[0], "name", "") in {
            "const_model_bnorm_spec_noquant_fused_batch_norm_v31",
            "const_const_fold_opt__1796",
        }
    }
    assert constant_permute_inputs == set()
    constant_reshape_inputs = {
        getattr(node.args[0], "name", "")
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.reshape.default"
        and getattr(node.args[0], "name", "") in {
            "const_model_mel_spec1_stft_hann_window_sub_2",
            "const_model_mel_spec2_stft_hann_window_sub_2",
            "const_model_bnorm_post_noquant_fused_batch_norm_v31",
            "const_const_fold_opt__1698",
        }
    }
    assert constant_reshape_inputs == set()
    early_swish_front_side_permute_sources = {
        getattr(node.args[0], "name", "")
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.permute.default"
        and getattr(node.args[0], "name", "") in {
            "conv2d_2",
            "conv2d_3",
            "conv2d_5",
            "conv2d_6",
        }
    }
    assert early_swish_front_side_permute_sources == set()
    early_residual_permute_sources = {
        getattr(node.args[0], "name", "")
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.permute.default"
        and getattr(node.args[0], "name", "") in {
            "conv2d_4",
            "conv2d_7",
            "add_2",
            "conv2d_10",
            "add_3",
        }
    }
    assert early_residual_permute_sources == set()
    early_swish_mul_inputs = {
        node.name: [getattr(arg, "name", "") for arg in node.args]
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.mul.Tensor"
        and node.name in {"mul_6", "mul_7", "mul_8", "mul_9"}
    }
    assert early_swish_mul_inputs == {
        "mul_6": ["conv2d_2", "sigmoid"],
        "mul_7": ["conv2d_3", "sigmoid_1"],
        "mul_8": ["conv2d_5", "sigmoid_2"],
        "mul_9": ["conv2d_6", "sigmoid_3"],
    }
    se_permute_bridge_sources = {
        getattr(node.args[0], "name", "")
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.permute.default"
        and getattr(node.args[0], "name", "") in {
            "reshape_8",
            "mul_13",
            "sigmoid_9",
            "mul_15",
        }
    }
    assert se_permute_bridge_sources == set()
    se_stage_inputs = {
        node.name: [getattr(arg, "name", "") for arg in node.args]
        for node in exported_program.module().graph.nodes
        if node.name in {"conv2d_13", "mul_15", "conv2d_15"}
        or str(node.target) in {"aten.mean.dim", "aten.reshape.default"}
    }
    mean_node_names = {
        node.name
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.mean.dim"
    }
    first_se_reshape_name = se_stage_inputs["conv2d_13"][0]
    assert first_se_reshape_name.startswith("reshape_")
    first_se_mean_name = se_stage_inputs[first_se_reshape_name][0]
    assert first_se_mean_name in mean_node_names
    assert se_stage_inputs[first_se_mean_name][0].startswith("mul_")
    assert se_stage_inputs["mul_15"][0].startswith("mul_")
    assert se_stage_inputs["mul_15"][1].startswith("sigmoid")
    assert se_stage_inputs["conv2d_15"][0] == "mul_15"
    assert all(
        getattr(node.args[0], "name", "") not in mean_node_names
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.mean.dim"
    )
    post_bn_permute_bridge_sources = {
        getattr(node.args[0], "name", "")
        for node in exported_program.module().graph.nodes
        if node.op == "call_function"
        and str(node.target) == "aten.permute.default"
        and getattr(node.args[0], "name", "") in {"add_13", "relu_1"}
    }
    assert post_bn_permute_bridge_sources == set()
    post_bn_stage_inputs = {
        node.name: [getattr(arg, "name", "") for arg in node.args]
        for node in exported_program.module().graph.nodes
        if node.name in {"add_13", "mul_64", "add_14", "relu_1", "conv2d_76"}
    }
    assert post_bn_stage_inputs["mul_64"][0] == "add_13"
    assert post_bn_stage_inputs["add_14"][0] == "mul_64"
    assert post_bn_stage_inputs["relu_1"] == ["add_14"]
    assert post_bn_stage_inputs["conv2d_76"][0] == "relu_1"
    reshape_node_names = {
        node.name
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.reshape.default"
    }
    assert all(
        getattr(node.args[0], "name", "") not in reshape_node_names
        for node in exported_program.module().graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.reshape.default"
    )

    pytorch_report = evaluate_pytorch_package_outputs(
        onnx_graph=onnx.load(str(model_path)),
        package_dir=str(package_path),
        output_report_path=str(output_dir / "birdnet_pytorch_accuracy_report.json"),
        num_samples=1,
    )
    assert pytorch_report["evaluation_pass"] is True


def test_convert_flatbuffer_direct_lstm_undefined_dim_outputs_all_native_artifacts_when_model_is_available(tmp_path) -> None:
    model_path = Path("lstm_undefined_dim.onnx")
    if not model_path.exists():
        pytest.skip("lstm_undefined_dim.onnx is not available")

    output_dir = tmp_path / "lstm_undefined_dim_out"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        flatbuffer_direct_output_torchscript=True,
        flatbuffer_direct_output_dynamo_onnx=True,
        flatbuffer_direct_output_exported_program=True,
        keep_shape_absolutely_input_names=["input"],
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "lstm_undefined_dim_pytorch"
    metadata = json.loads((package_path / "metadata.json").read_text())
    assert metadata["execution_backend"] == "native"
    assert (package_path / "lstm_undefined_dim_float32_jit.pt").exists()
    assert (package_path / "lstm_undefined_dim_float32_dynamo.onnx").exists()
    assert (package_path / "lstm_undefined_dim_float32_ep.pt2").exists()
    assert "error" not in metadata["torchscript"]
    assert "error" not in metadata["dynamo_onnx"]
    assert "error" not in metadata["exported_program"]


def test_convert_flatbuffer_direct_recognizer_japanese_g2_outputs_all_native_artifacts_when_model_is_available(
    tmp_path,
) -> None:
    model_path = Path("recognizer_japanese_g2.onnx")
    if not model_path.exists():
        pytest.skip("recognizer_japanese_g2.onnx is not available")

    output_dir = tmp_path / "recognizer_japanese_g2_out"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        flatbuffer_direct_output_torchscript=True,
        flatbuffer_direct_output_dynamo_onnx=True,
        flatbuffer_direct_output_exported_program=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "recognizer_japanese_g2_pytorch"
    metadata = json.loads((package_path / "metadata.json").read_text())
    assert metadata["execution_backend"] == "native"
    assert (package_path / "recognizer_japanese_g2_jit.pt").exists()
    assert (package_path / "recognizer_japanese_g2_dynamo.onnx").exists()
    assert (package_path / "recognizer_japanese_g2_ep.pt2").exists()
    assert "error" not in metadata["torchscript"]
    assert "error" not in metadata["dynamo_onnx"]
    assert "error" not in metadata["exported_program"]
    assert "skipped_reason" not in metadata["torchscript"]
    assert "skipped_reason" not in metadata["dynamo_onnx"]
    assert "skipped_reason" not in metadata["exported_program"]


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


def test_export_artifacts_handle_maximum_minimum_scalar_constants(tmp_path) -> None:
    model_ir = ModelIR(name="maximum_minimum_scalar_constants")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 1],
        shape_signature=[1, 2, 3, 1],
    )
    model_ir.tensors["lo"] = TensorIR(
        name="lo",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([-1.0], dtype=np.float32),
    )
    model_ir.tensors["hi"] = TensorIR(
        name="hi",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0], dtype=np.float32),
    )
    model_ir.tensors["mid"] = TensorIR(
        name="mid",
        dtype="FLOAT32",
        shape=[1, 2, 3, 1],
        shape_signature=[1, 2, 3, 1],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 1],
        shape_signature=[1, 2, 3, 1],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="MAXIMUM",
                inputs=["x", "lo"],
                outputs=["mid"],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="MINIMUM",
                inputs=["mid", "hi"],
                outputs=["y"],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "maximum_minimum_scalar_constants_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.maximum(" in model_source
    assert "torch.minimum(" in model_source
    assert "torch.maximum(x, -1.0)" not in model_source
    assert "torch.minimum(mid, 1.0)" not in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor(
        [[[[-2.0], [-0.5], [0.5]], [[1.5], [2.0], [-3.0]]]],
        dtype=torch.float32,
    )
    out = model(x)
    assert torch.allclose(out, torch.clamp(x, min=-1.0, max=1.0))

    torchscript_path = export_torchscript_from_generated_package(package_dir=package_path)
    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert torchscript_path is not None
    assert Path(torchscript_path).exists()
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()


def test_export_pytorch_package_channel_last_prelu_supports_dynamo_and_exported_program(tmp_path) -> None:
    model_ir = ModelIR(name="channel_last_prelu")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 256, 256, 8],
        shape_signature=[1, 256, 256, 8],
        logical_layout="NHWC",
    )
    model_ir.tensors["alpha"] = TensorIR(
        name="alpha",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.asarray([0.1] * 8, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 256, 256, 8],
        shape_signature=[1, 256, 256, 8],
        logical_layout="NHWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="PRELU",
            inputs=["x", "alpha"],
            outputs=["y"],
            options={},
        )
    )

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "channel_last_prelu_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.prelu_0(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()" in model_source

    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.randn(1, 256, 256, 8)
    out = model(x)
    alpha = torch.tensor([0.1] * 8, dtype=torch.float32)
    ref = F.prelu(x.permute(0, 3, 1, 2).contiguous(), alpha).permute(0, 2, 3, 1).contiguous()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    dynamo_onnx_path = export_dynamo_onnx_from_generated_package(package_dir=package_path)
    exported_program_path = export_exported_program_from_generated_package(package_dir=package_path)
    assert dynamo_onnx_path is not None
    assert Path(dynamo_onnx_path).exists()
    assert exported_program_path is not None
    assert Path(exported_program_path).exists()
