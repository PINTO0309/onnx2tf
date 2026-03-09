import builtins
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

import onnx2tf
from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.pytorch_exporter import (
    ModelIRPyTorchExportError,
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
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


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


def test_export_pytorch_package_roundtrip_add_relu(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_relu_model_ir(),
        output_folder_path=str(tmp_path / "add_relu_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    y = torch.tensor([[4.0, 5.0, -10.0]], dtype=torch.float32)
    out = model(x, y)
    assert torch.allclose(out, torch.relu(x + y))
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["outputs"] == ["z"]


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
