import builtins
import json
import os
import subprocess
import tempfile
from typing import Any

import flatbuffers
import numpy as np
import onnx
import onnx2tf
import pytest
import tensorflow as tf
import tf_keras
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.saved_model_exporter import (
    ModelIRSavedModelExportError,
    export_saved_model_from_model_ir,
    get_known_model_ir_op_types,
    get_supported_kernel_op_types,
)
from onnx2tf.tflite_builder.tflite_importer import import_model_ir_from_tflite
from onnx2tf.tflite_builder.schema_loader import load_schema_module
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.split_planner import (
    crop_model_ir_by_boundary_tensors,
    rewrite_model_ir_disable_group_convolution,
    rewrite_model_ir_unfold_batchmatmul,
    rewrite_model_ir_unroll_recurrent_ops,
)

Interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter


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
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ADD",
            inputs=["x", "y"],
            outputs=["z"],
            options={},
        )
    )
    return model_ir


def _make_add_relu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    sum_tensor = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    add = helper.make_node("Add", ["x", "y"], ["sum"], name="AddNode")
    relu = helper.make_node("Relu", ["sum"], ["z"], name="ReluNode")
    graph = helper.make_graph(
        [add, relu],
        "add_relu_graph",
        [x, y],
        [z],
        value_info=[sum_tensor],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_batched_matmul_model() -> onnx.ModelProto:
    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [2, 3, 4])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [2, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 5])
    node = helper.make_node("MatMul", ["lhs", "rhs"], ["y"], name="MatMulNode")
    graph = helper.make_graph([node], "batched_matmul_graph", [lhs, rhs], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_simple_rnn_model() -> onnx.ModelProto:
    rng = np.random.default_rng(4)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 1, 1, 2])
    y_h = helper.make_tensor_value_info("y_h", TensorProto.FLOAT, [1, 1, 2])
    w_init = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [1, 2, 2],
        rng.standard_normal((1, 2, 2)).astype(np.float32).reshape(-1).tolist(),
    )
    r_init = helper.make_tensor(
        "r",
        TensorProto.FLOAT,
        [1, 2, 2],
        rng.standard_normal((1, 2, 2)).astype(np.float32).reshape(-1).tolist(),
    )
    b_init = helper.make_tensor(
        "b",
        TensorProto.FLOAT,
        [1, 4],
        rng.standard_normal((1, 4)).astype(np.float32).reshape(-1).tolist(),
    )
    node = helper.make_node(
        "RNN",
        ["x", "w", "r", "b"],
        ["y", "y_h"],
        name="RnnNode",
        hidden_size=2,
    )
    graph = helper.make_graph(
        [node],
        "simple_rnn_graph",
        [x],
        [y, y_h],
        initializer=[w_init, r_init, b_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])


def _make_add_relu_model_ir() -> ModelIR:
    model_ir = ModelIR(name="add_relu_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.tensors["sum"] = TensorIR(
        name="sum",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ADD",
            inputs=["x", "y"],
            outputs=["sum"],
            options={},
        )
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["sum"],
            outputs=["z"],
            options={},
        )
    )
    return model_ir


def _save_model(tmpdir: str, name: str, model: onnx.ModelProto) -> str:
    model_path = os.path.join(tmpdir, f"{name}.onnx")
    onnx.save(model, model_path)
    return model_path


def _write_model_ir_as_tflite(tmpdir: str, name: str, model_ir: ModelIR) -> str:
    schema_tflite = load_schema_module(tmpdir)
    tflite_path = os.path.join(tmpdir, f"{name}.tflite")
    write_model_file(
        schema_tflite=schema_tflite,
        model_ir=model_ir,
        output_tflite_path=tflite_path,
    )
    return tflite_path


def _mutate_tflite_model_in_place(
    *,
    tflite_path: str,
    output_dir: str,
    mutator,
) -> None:
    schema_tflite = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model_obj = schema_tflite["ModelT"].InitFromObj(
        schema_tflite["Model"].GetRootAs(model_bytes, 0)
    )
    mutator(model_obj)
    builder = flatbuffers.Builder()
    model_offset = model_obj.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    with open(tflite_path, "wb") as f:
        f.write(bytes(builder.Output()))


def _run_saved_model_single_output(
    saved_model_path: str,
    input_name: str,
    input_value: np.ndarray,
    output_name: str,
) -> np.ndarray:
    module: Any = tf.saved_model.load(saved_model_path)
    assert module is not None
    fn = module.signatures["serving_default"]
    outputs = fn(**{input_name: tf.constant(input_value)})
    return np.asarray(outputs[output_name].numpy())


def _run_saved_model_with_inputs(
    saved_model_path: str,
    inputs: dict[str, np.ndarray],
) -> tuple[list[str], dict[str, np.ndarray]]:
    module: Any = tf.saved_model.load(saved_model_path)
    assert module is not None
    fn = module.signatures["serving_default"]
    outputs = fn(
        **{
            name: tf.constant(value)
            for name, value in inputs.items()
        }
    )
    return (
        list(fn.structured_input_signature[1].keys()),
        {
            str(name): np.asarray(value.numpy())
            for name, value in outputs.items()
        },
    )


def _run_tflite_single_output(
    tflite_path: str,
    input_value: np.ndarray,
) -> np.ndarray:
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_value)
    interpreter.invoke()
    return np.asarray(interpreter.get_tensor(output_details[0]["index"]))


def _run_tflite_with_inputs(
    tflite_path: str,
    input_values: list[np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    assert len(input_values) == len(input_details)
    for detail, value in zip(input_details, input_values):
        interpreter.set_tensor(detail["index"], np.asarray(value))
    interpreter.invoke()
    outputs = [
        np.asarray(interpreter.get_tensor(detail["index"]))
        for detail in output_details
    ]
    return input_details, output_details, outputs


def _disable_tf_converter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise AssertionError("tf_converter fallback should not run")

    monkeypatch.setattr(
        tf.lite.TFLiteConverter,
        "from_keras_model",
        staticmethod(_raise),
    )
    monkeypatch.setattr(
        tf.lite.TFLiteConverter,
        "from_concrete_functions",
        staticmethod(_raise),
    )


def _assert_dir_empty_or_missing(path: str) -> None:
    assert not os.path.exists(path) or len(os.listdir(path)) == 0


def test_flatbuffer_direct_not_use_onnxsim_skips_onnxsim_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_check_output = subprocess.check_output

    def _guard_check_output(*args, **kwargs):
        cmd = args[0] if len(args) > 0 else kwargs.get("args")
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and str(cmd[0]) == "onnxsim":
            raise AssertionError("onnxsim should not run when not_use_onnxsim=True")
        return original_check_output(*args, **kwargs)

    _disable_tf_converter_fallback(monkeypatch)
    monkeypatch.setattr(subprocess, "check_output", _guard_check_output)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_no_onnxsim", _make_add_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            disable_strict_mode=True,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            not_use_onnxsim=True,
        )
        assert os.path.exists(os.path.join(output_dir, "add_no_onnxsim_float32.tflite"))


@pytest.mark.parametrize(
    "kwargs",
    [
        (
            {
                "tflite_backend": "tf_converter",
                "flatbuffer_direct_output_saved_model": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "flatbuffer_direct_output_saved_model": True,
                "disable_model_save": True,
            }
        ),
    ],
)
def test_flatbuffer_direct_output_saved_model_validation(
    kwargs: dict,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add", _make_add_model())
        with pytest.raises(SystemExit):
            onnx2tf.convert(
                input_onnx_file_path=model_path,
                output_folder_path=tmpdir,
                verbosity="error",
                disable_strict_mode=True,
                **kwargs,
            )


def test_flatbuffer_direct_cotof_without_fdosm_skips_saved_model_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_skip_sm_check", _make_add_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            check_onnx_tf_outputs_elementwise_close_full=True,
        )

        report_path = os.path.join(
            tmpdir,
            "add_skip_sm_check_saved_model_validation_report.json",
        )
        assert not os.path.exists(report_path)
        captured = capsys.readouterr()
        assert "SavedModel inference check skipped because SavedModel path is unavailable" not in (
            captured.out + captured.err
        )


def test_saved_model_exporter_add_smoke() -> None:
    model_ir = ModelIR(name="add_smoke")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[-1, 3],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ADD",
            inputs=["x", "y"],
            outputs=["z"],
            options={},
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = export_saved_model_from_model_ir(
            model_ir=model_ir,
            output_folder_path=tmpdir,
        )
        assert os.path.exists(os.path.join(saved_model_path, "saved_model.pb"))
        module: Any = tf.saved_model.load(saved_model_path)
        assert module is not None
        fn = module.signatures["serving_default"]
        outputs = fn(
            x=tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32),
            y=tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float32),
        )
        np.testing.assert_allclose(
            outputs["z"].numpy(),
            np.asarray([[5.0, 7.0, 9.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_tflite_importer_add_roundtrip_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_roundtrip", model_ir)
        imported = import_model_ir_from_tflite(
            tflite_file_path=tflite_path,
            output_folder_path=tmpdir,
        )

    assert imported.inputs == ["x", "y"]
    assert imported.outputs == ["z"]
    assert "x" in imported.tensors
    assert imported.tensors["x"].dtype == "FLOAT32"
    assert imported.tensors["x"].shape == [1, 3]
    assert imported.operators[0].op_type == "ADD"
    assert imported.operators[0].inputs == ["x", "y"]
    assert imported.operators[0].outputs == ["z"]


def test_tflite_importer_signature_name_priority() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_signature_priority", model_ir)

        def _mutator(model_obj) -> None:
            signature = model_obj.signatureDefs[0]
            signature.inputs[0].name = "sig_x"
            signature.inputs[1].name = "sig_y"
            signature.outputs[0].name = "sig_z"

        _mutate_tflite_model_in_place(
            tflite_path=tflite_path,
            output_dir=tmpdir,
            mutator=_mutator,
        )
        imported = import_model_ir_from_tflite(
            tflite_file_path=tflite_path,
            output_folder_path=tmpdir,
        )

    assert imported.inputs == ["sig_x", "sig_y"]
    assert imported.outputs == ["sig_z"]
    assert "sig_x" in imported.tensors
    assert "sig_y" in imported.tensors
    assert "sig_z" in imported.tensors


def test_tflite_importer_normalizes_empty_duplicate_tensor_names() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_empty_tensor_names", model_ir)

        def _mutator(model_obj) -> None:
            for tensor in model_obj.subgraphs[0].tensors:
                tensor.name = ""
            if model_obj.signatureDefs and len(model_obj.signatureDefs) > 0:
                for tensor_map in model_obj.signatureDefs[0].inputs:
                    tensor_map.name = ""
                for tensor_map in model_obj.signatureDefs[0].outputs:
                    tensor_map.name = ""

        _mutate_tflite_model_in_place(
            tflite_path=tflite_path,
            output_dir=tmpdir,
            mutator=_mutator,
        )
        imported = import_model_ir_from_tflite(
            tflite_file_path=tflite_path,
            output_folder_path=tmpdir,
        )

    tensor_names = list(imported.tensors.keys())
    assert len(tensor_names) == len(set(tensor_names))
    assert all(name.strip() != "" for name in tensor_names)
    assert all(name.startswith("sg0_tensor") for name in imported.inputs + imported.outputs)


def test_crop_model_ir_by_boundary_tensors_rejects_nested_subgraph_tensor_name() -> None:
    model_ir = _make_add_relu_model_ir()
    nested = ModelIR(name="nested")
    nested.tensors["inner_tensor"] = TensorIR(
        name="inner_tensor",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.subgraphs = [nested]
    with pytest.raises(ValueError, match="nested subgraph tensor names are unsupported"):
        crop_model_ir_by_boundary_tensors(
            model_ir=model_ir,
            requested_inputs=["inner_tensor"],
            requested_outputs=["z"],
        )


def test_crop_model_ir_by_boundary_tensors_rejects_unreachable_outputs() -> None:
    model_ir = _make_add_relu_model_ir()
    with pytest.raises(ValueError, match="requested outputs are not reachable"):
        crop_model_ir_by_boundary_tensors(
            model_ir=model_ir,
            requested_inputs=["y"],
            requested_outputs=["sum"],
        )


def _make_while_model_ir() -> ModelIR:
    model_ir = ModelIR(name="while_model")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="INT32", shape=[], shape_signature=[])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="INT32", shape=[], shape_signature=[])

    cond_ir = ModelIR(name="while_cond")
    cond_ir.inputs = ["cond_in"]
    cond_ir.outputs = ["cond_out"]
    cond_ir.tensors["cond_in"] = TensorIR(name="cond_in", dtype="INT32", shape=[], shape_signature=[])
    cond_ir.tensors["limit"] = TensorIR(
        name="limit",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(5, dtype=np.int32),
    )
    cond_ir.tensors["cond_out"] = TensorIR(name="cond_out", dtype="BOOL", shape=[], shape_signature=[])
    cond_ir.operators.append(
        OperatorIR(
            op_type="LESS",
            inputs=["cond_in", "limit"],
            outputs=["cond_out"],
            options={},
        )
    )

    body_ir = ModelIR(name="while_body")
    body_ir.inputs = ["body_in"]
    body_ir.outputs = ["body_out"]
    body_ir.tensors["body_in"] = TensorIR(name="body_in", dtype="INT32", shape=[], shape_signature=[])
    body_ir.tensors["one"] = TensorIR(
        name="one",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1, dtype=np.int32),
    )
    body_ir.tensors["body_out"] = TensorIR(name="body_out", dtype="INT32", shape=[], shape_signature=[])
    body_ir.operators.append(
        OperatorIR(
            op_type="ADD",
            inputs=["body_in", "one"],
            outputs=["body_out"],
            options={},
        )
    )

    model_ir.subgraphs = [cond_ir, body_ir]
    model_ir.operators.append(
        OperatorIR(
            op_type="WHILE",
            inputs=["x"],
            outputs=["y"],
            options={
                "condSubgraphIndex": 1,
                "bodySubgraphIndex": 2,
            },
        )
    )
    return model_ir


def _make_unidirectional_sequence_rnn_model_ir() -> ModelIR:
    rng = np.random.default_rng(0)
    time_steps = 3
    batch = 1
    input_size = 2
    hidden_size = 2

    model_ir = ModelIR(name="sequence_rnn")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[time_steps, batch, input_size],
        shape_signature=[time_steps, batch, input_size],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[hidden_size, input_size],
        shape_signature=[hidden_size, input_size],
        data=rng.standard_normal((hidden_size, input_size)).astype(np.float32),
    )
    model_ir.tensors["r"] = TensorIR(
        name="r",
        dtype="FLOAT32",
        shape=[hidden_size, hidden_size],
        shape_signature=[hidden_size, hidden_size],
        data=rng.standard_normal((hidden_size, hidden_size)).astype(np.float32),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[hidden_size],
        shape_signature=[hidden_size],
        data=rng.standard_normal((hidden_size,)).astype(np.float32),
    )
    model_ir.tensors["h0"] = TensorIR(
        name="h0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
        data=np.zeros((batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[time_steps, batch, hidden_size],
        shape_signature=[time_steps, batch, hidden_size],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="UNIDIRECTIONAL_SEQUENCE_RNN",
            inputs=["x", "w", "r", "b", "h0"],
            outputs=["y"],
            options={
                "timeMajor": True,
                "fusedActivationFunction": "TANH",
                "asymmetricQuantizeInputs": False,
            },
        )
    )
    return model_ir


def _make_unidirectional_sequence_lstm_model_ir() -> ModelIR:
    rng = np.random.default_rng(1)
    time_steps = 3
    batch = 1
    input_size = 2
    hidden_size = 2

    model_ir = ModelIR(name="sequence_lstm_uni")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[time_steps, batch, input_size],
        shape_signature=[time_steps, batch, input_size],
    )
    for name in ["wi", "wf", "wc", "wo"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[hidden_size, input_size],
            shape_signature=[hidden_size, input_size],
            data=rng.standard_normal((hidden_size, input_size)).astype(np.float32),
        )
    for name in ["ri", "rf", "rc", "ro"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[hidden_size, hidden_size],
            shape_signature=[hidden_size, hidden_size],
            data=rng.standard_normal((hidden_size, hidden_size)).astype(np.float32),
        )
    for name in ["bi", "bf", "bc", "bo"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[hidden_size],
            shape_signature=[hidden_size],
            data=rng.standard_normal((hidden_size,)).astype(np.float32),
        )
    model_ir.tensors["h0"] = TensorIR(
        name="h0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
        data=np.zeros((batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["c0"] = TensorIR(
        name="c0",
        dtype="FLOAT32",
        shape=[batch, hidden_size],
        shape_signature=[batch, hidden_size],
        data=np.zeros((batch, hidden_size), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[time_steps, batch, hidden_size],
        shape_signature=[time_steps, batch, hidden_size],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="UNIDIRECTIONAL_SEQUENCE_LSTM",
            inputs=[
                "x", "wi", "wf", "wc", "wo",
                "ri", "rf", "rc", "ro",
                "", "", "",
                "bi", "bf", "bc", "bo",
                "", "",
                "h0", "c0",
                "", "", "", "",
            ],
            outputs=["y"],
            options={
                "fusedActivationFunction": "TANH",
                "cellClip": 0.0,
                "projClip": 0.0,
                "timeMajor": True,
                "asymmetricQuantizeInputs": False,
                "diagonalRecurrentTensors": False,
            },
        )
    )
    return model_ir


def _make_bidirectional_sequence_lstm_model_ir() -> ModelIR:
    rng = np.random.default_rng(2)
    time_steps = 3
    batch = 1
    input_size = 2
    hidden_size = 2

    model_ir = ModelIR(name="sequence_lstm_bi")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[time_steps, batch, input_size],
        shape_signature=[time_steps, batch, input_size],
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


def _make_grouped_conv_model_ir() -> ModelIR:
    rng = np.random.default_rng(3)
    model_ir = ModelIR(name="grouped_conv")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 4],
        shape_signature=[1, 4, 4, 4],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[4, 1, 1, 2],
        shape_signature=[4, 1, 1, 2],
        data=rng.standard_normal((4, 1, 1, 2)).astype(np.float32),
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
        shape=[1, 4, 4, 4],
        shape_signature=[1, 4, 4, 4],
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
                "fusedActivationFunction": "RELU",
            },
        )
    )
    return model_ir


def _make_batched_matmul_model_ir() -> ModelIR:
    model_ir = ModelIR(name="batched_matmul")
    model_ir.inputs = ["lhs", "rhs"]
    model_ir.outputs = ["y"]
    model_ir.tensors["lhs"] = TensorIR(
        name="lhs",
        dtype="FLOAT32",
        shape=[2, 3, 4],
        shape_signature=[2, 3, 4],
    )
    model_ir.tensors["rhs"] = TensorIR(
        name="rhs",
        dtype="FLOAT32",
        shape=[2, 4, 5],
        shape_signature=[2, 4, 5],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[2, 3, 5],
        shape_signature=[2, 3, 5],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs", "rhs"],
            outputs=["y"],
            options={"adjX": False, "adjY": False},
        )
    )
    return model_ir


def test_saved_model_exporter_while_matches_tflite() -> None:
    model_ir = _make_while_model_ir()
    input_data = np.asarray(2, dtype=np.int32)
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = export_saved_model_from_model_ir(
            model_ir=model_ir,
            output_folder_path=os.path.join(tmpdir, "sm"),
        )
        tflite_path = _write_model_ir_as_tflite(tmpdir, "while", model_ir)
        sm_out = _run_saved_model_single_output(saved_model_path, "x", input_data, "y")
        tflite_out = _run_tflite_single_output(tflite_path, input_data)
        np.testing.assert_allclose(sm_out, tflite_out, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "builder,seed",
    [
        (_make_unidirectional_sequence_rnn_model_ir, 10),
        (_make_unidirectional_sequence_lstm_model_ir, 11),
        (_make_bidirectional_sequence_lstm_model_ir, 12),
    ],
)
def test_saved_model_exporter_recurrent_smoke(
    builder,
    seed: int,
) -> None:
    model_ir = builder()
    rng = np.random.default_rng(seed)
    x_shape = model_ir.tensors["x"].shape
    input_data = rng.standard_normal(x_shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = export_saved_model_from_model_ir(
            model_ir=model_ir,
            output_folder_path=os.path.join(tmpdir, "sm"),
        )
        sm_out = _run_saved_model_single_output(saved_model_path, "x", input_data, "y")
        expected_shape = tuple(int(v) for v in model_ir.tensors["y"].shape)
        assert tuple(sm_out.shape) == expected_shape
        assert np.isfinite(sm_out).all()


def test_rewrite_model_ir_disable_group_convolution_smoke() -> None:
    model_ir = _make_grouped_conv_model_ir()
    rewritten_model_ir, rewritten_count = rewrite_model_ir_disable_group_convolution(
        model_ir=model_ir,
    )
    assert rewritten_count == 1
    op_types = [str(op.op_type) for op in rewritten_model_ir.operators]
    assert op_types.count("SPLIT") == 1
    assert op_types.count("CONV_2D") == 2
    assert op_types.count("CONCATENATION") == 1
    concat_op = next(
        op for op in rewritten_model_ir.operators
        if str(op.op_type) == "CONCATENATION"
    )
    assert str(concat_op.options.get("fusedActivationFunction", "NONE")).upper() == "RELU"


def test_rewrite_model_ir_unfold_batchmatmul_preserves_saved_model_outputs() -> None:
    model_ir = _make_batched_matmul_model_ir()
    rewritten_model_ir, rewritten_count = rewrite_model_ir_unfold_batchmatmul(
        model_ir=model_ir,
    )
    assert rewritten_count == 1
    assert "CONCATENATION" in [str(op.op_type) for op in rewritten_model_ir.operators]
    rng = np.random.default_rng(13)
    lhs = rng.standard_normal((2, 3, 4)).astype(np.float32)
    rhs = rng.standard_normal((2, 4, 5)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        original_sm = export_saved_model_from_model_ir(
            model_ir=model_ir,
            output_folder_path=os.path.join(tmpdir, "orig"),
        )
        rewritten_sm = export_saved_model_from_model_ir(
            model_ir=rewritten_model_ir,
            output_folder_path=os.path.join(tmpdir, "rewritten"),
        )
        original = _run_saved_model_with_inputs(
            original_sm,
            {"lhs": lhs, "rhs": rhs},
        )[1]["y"]
        rewritten = _run_saved_model_with_inputs(
            rewritten_sm,
            {"lhs": lhs, "rhs": rhs},
        )[1]["y"]
        np.testing.assert_allclose(original, rewritten, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "builder,seed",
    [
        (_make_unidirectional_sequence_rnn_model_ir, 21),
        (_make_unidirectional_sequence_lstm_model_ir, 22),
        (_make_bidirectional_sequence_lstm_model_ir, 23),
    ],
)
def test_rewrite_model_ir_unroll_recurrent_ops_preserves_saved_model_outputs(
    builder,
    seed: int,
) -> None:
    model_ir = builder()
    rewritten_model_ir, rewritten_count = rewrite_model_ir_unroll_recurrent_ops(
        model_ir=model_ir,
    )
    assert rewritten_count == 1
    rewritten_op_types = [str(op.op_type) for op in rewritten_model_ir.operators]
    assert "UNIDIRECTIONAL_SEQUENCE_RNN" not in rewritten_op_types
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" not in rewritten_op_types
    assert "BIDIRECTIONAL_SEQUENCE_LSTM" not in rewritten_op_types
    assert "BATCH_MATMUL" in rewritten_op_types
    rng = np.random.default_rng(seed)
    input_data = rng.standard_normal(tuple(model_ir.tensors["x"].shape)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        original_sm = export_saved_model_from_model_ir(
            model_ir=model_ir,
            output_folder_path=os.path.join(tmpdir, "orig"),
        )
        rewritten_sm = export_saved_model_from_model_ir(
            model_ir=rewritten_model_ir,
            output_folder_path=os.path.join(tmpdir, "rewritten"),
        )
        original = _run_saved_model_single_output(original_sm, "x", input_data, "y")
        rewritten = _run_saved_model_single_output(rewritten_sm, "x", input_data, "y")
        np.testing.assert_allclose(original, rewritten, rtol=1e-5, atol=1e-5)


def test_rewrite_model_ir_unroll_recurrent_ops_rewires_state_aliases() -> None:
    model_ir = _make_unidirectional_sequence_rnn_model_ir()
    model_ir.outputs = ["y_h"]
    model_ir.tensors["y_h"] = TensorIR(
        name="y_h",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["h0", "reshape_shape"],
            outputs=["y_h"],
            options={"newShape": [1, 2]},
        )
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 2], dtype=np.int32),
    )
    rewritten_model_ir, rewritten_count = rewrite_model_ir_unroll_recurrent_ops(
        model_ir=model_ir,
    )
    assert rewritten_count == 1
    reshape_op = next(op for op in rewritten_model_ir.operators if str(op.op_type) == "RESHAPE")
    assert str(reshape_op.inputs[0]) != "h0"


def test_tflite_direct_input_disable_group_convolution_smoke() -> None:
    model_ir = _make_grouped_conv_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "grouped_conv_direct_input", model_ir)
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            disable_group_convolution=True,
        )
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))


def test_tflite_direct_input_enable_batchmatmul_unfold_smoke() -> None:
    model_ir = _make_batched_matmul_model_ir()
    rng = np.random.default_rng(24)
    lhs = rng.standard_normal((2, 3, 4)).astype(np.float32)
    rhs = rng.standard_normal((2, 4, 5)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "batched_matmul_direct_input", model_ir)
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            enable_batchmatmul_unfold=True,
        )
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))
        _, outputs = _run_saved_model_with_inputs(
            output_dir,
            {"lhs": lhs, "rhs": rhs},
        )
        np.testing.assert_allclose(outputs["y"], np.matmul(lhs, rhs), rtol=1e-5, atol=1e-5)


def test_tflite_direct_input_enable_rnn_unroll_smoke() -> None:
    model_ir = _make_unidirectional_sequence_rnn_model_ir()
    rng = np.random.default_rng(25)
    x = rng.standard_normal(tuple(model_ir.tensors["x"].shape)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "sequence_rnn_direct_input", model_ir)
        reference_saved_model_path = export_saved_model_from_model_ir(
            model_ir=model_ir,
            output_folder_path=os.path.join(tmpdir, "reference_sm"),
        )
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            enable_rnn_unroll=True,
        )
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))
        saved_model_output = _run_saved_model_single_output(output_dir, "x", x, "y")
        reference_output = _run_saved_model_single_output(
            reference_saved_model_path,
            "x",
            x,
            "y",
        )
        np.testing.assert_allclose(saved_model_output, reference_output, rtol=1e-5, atol=1e-5)


def test_saved_model_exporter_custom_op_is_rejected() -> None:
    model_ir = ModelIR(name="custom_ng")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.operators.append(
        OperatorIR(
            op_type="CUSTOM",
            inputs=["x"],
            outputs=["y"],
            options={
                "customCode": "MyCustom",
                "onnxOp": "CustomAdd",
                "onnxNodeName": "CustomAddNode",
            },
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ModelIRSavedModelExportError, match="CUSTOM ops"):
            export_saved_model_from_model_ir(
                model_ir=model_ir,
                output_folder_path=tmpdir,
            )


def test_flatbuffer_direct_output_saved_model_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add", _make_add_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_saved_model=True,
        )
        assert os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
        assert os.path.exists(os.path.join(tmpdir, "add_float32.tflite"))


def test_flatbuffer_direct_enable_batchmatmul_unfold_smoke() -> None:
    rng = np.random.default_rng(26)
    lhs = rng.standard_normal((2, 3, 4)).astype(np.float32)
    rhs = rng.standard_normal((2, 4, 5)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "batched_matmul", _make_batched_matmul_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_saved_model=True,
            enable_batchmatmul_unfold=True,
        )
        assert os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
        _, outputs = _run_saved_model_with_inputs(
            tmpdir,
            {"lhs": lhs, "rhs": rhs},
        )
        np.testing.assert_allclose(outputs["y"], np.matmul(lhs, rhs), rtol=1e-5, atol=1e-5)


def test_flatbuffer_direct_enable_rnn_unroll_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "simple_rnn", _make_simple_rnn_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_saved_model=True,
            enable_rnn_unroll=True,
        )
        assert os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
        assert os.path.exists(os.path.join(tmpdir, "simple_rnn_float32.tflite"))


def test_flatbuffer_direct_output_h5_without_tf_converter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tf_converter_fallback(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_h5_direct", _make_add_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            output_h5=True,
        )
        h5_path = os.path.join(output_dir, "add_h5_direct_float32.h5")
        assert os.path.exists(h5_path)
        loaded = tf_keras.models.load_model(
            h5_path,
            compile=False,
            safe_mode=False,
        )
        assert loaded is not None
        outputs = loaded(
            [
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            ]
        )
        np.testing.assert_allclose(
            np.asarray(outputs),
            np.asarray([[5.0, 7.0, 9.0]], dtype=np.float32),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_flatbuffer_direct_output_keras_v3_without_tf_converter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tf_converter_fallback(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_kv3_direct", _make_add_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            output_keras_v3=True,
        )
        keras_path = os.path.join(output_dir, "add_kv3_direct_float32_v3.keras")
        assert os.path.exists(keras_path)
        loaded = tf_keras.models.load_model(
            keras_path,
            compile=False,
            safe_mode=False,
        )
        assert loaded is not None
        outputs = loaded(
            [
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            ]
        )
        np.testing.assert_allclose(
            np.asarray(outputs),
            np.asarray([[5.0, 7.0, 9.0]], dtype=np.float32),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_flatbuffer_direct_output_tfv1_pb_without_tf_converter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tf_converter_fallback(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_pb_direct", _make_add_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            output_tfv1_pb=True,
        )
        assert os.path.exists(
            os.path.join(output_dir, "add_pb_direct_float32.pb")
        )


def test_flatbuffer_direct_disable_model_save_leaves_no_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_no_save", _make_add_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            disable_model_save=True,
        )
        _assert_dir_empty_or_missing(output_dir)


def test_flatbuffer_direct_interrupt_input_names_model_ir_crop_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_relu_inimc", _make_add_relu_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            input_names_to_interrupt_model_conversion=["sum"],
        )
        tflite_path = os.path.join(output_dir, "add_relu_inimc_float32.tflite")
        input_details, output_details, outputs = _run_tflite_with_inputs(
            tflite_path,
            [np.asarray([[-1.0, 2.0, -3.0]], dtype=np.float32)],
        )
        assert len(input_details) == 1
        assert len(output_details) == 1
        assert input_details[0]["shape"].tolist() == [1, 3]
        np.testing.assert_allclose(
            outputs[0],
            np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_flatbuffer_direct_interrupt_output_names_model_ir_crop_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_relu_onimc", _make_add_relu_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            output_names_to_interrupt_model_conversion=["sum"],
        )
        tflite_path = os.path.join(output_dir, "add_relu_onimc_float32.tflite")
        input_details, output_details, outputs = _run_tflite_with_inputs(
            tflite_path,
            [
                np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32),
                np.asarray([[4.0, 5.0, -6.0]], dtype=np.float32),
            ],
        )
        assert len(input_details) == 2
        assert len(output_details) == 1
        np.testing.assert_allclose(
            outputs[0],
            np.asarray([[5.0, 3.0, -3.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_flatbuffer_direct_interrupt_zero_op_crop_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_relu_zero", _make_add_relu_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            input_names_to_interrupt_model_conversion=["sum"],
            output_names_to_interrupt_model_conversion=["sum"],
        )
        tflite_path = os.path.join(output_dir, "add_relu_zero_float32.tflite")
        input_details, output_details, outputs = _run_tflite_with_inputs(
            tflite_path,
            [np.asarray([[7.0, -8.0, 9.0]], dtype=np.float32)],
        )
        assert len(input_details) == 1
        assert len(output_details) == 1
        np.testing.assert_allclose(
            outputs[0],
            np.asarray([[7.0, -8.0, 9.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_flatbuffer_direct_interrupt_with_enable_auto_split_model_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_relu_split_crop", _make_add_relu_model())
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=output_dir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            enable_auto_split_model=True,
            output_names_to_interrupt_model_conversion=["sum"],
        )
        assert os.path.exists(
            os.path.join(output_dir, "add_relu_split_crop_split_plan.json")
        )
        assert os.path.exists(
            os.path.join(output_dir, "add_relu_split_crop_split_manifest.json")
        )


def test_tflite_direct_input_saved_model_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input", model_ir)
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
        )
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))


def test_tflite_direct_input_without_fdopt_does_not_import_pytorch_exporters(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "onnx2tf.tflite_builder.pytorch_exporter",
            "onnx2tf.tflite_builder.split_pytorch_exporter",
        }:
            raise AssertionError(f"unexpected import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_no_fdopt", model_ir)
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
        )
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))


def test_tflite_direct_input_output_h5_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(
            tmpdir,
            "add_tflite_direct_input_h5",
            model_ir,
        )
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            output_h5=True,
        )
        h5_path = os.path.join(output_dir, "add_tflite_direct_input_h5_float32.h5")
        assert os.path.exists(h5_path)
        loaded = tf_keras.models.load_model(
            h5_path,
            compile=False,
            safe_mode=False,
        )
        assert loaded is not None
        outputs = loaded(
            [
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            ]
        )
        np.testing.assert_allclose(
            np.asarray(outputs),
            np.asarray([[5.0, 7.0, 9.0]], dtype=np.float32),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_tflite_direct_input_output_keras_v3_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(
            tmpdir,
            "add_tflite_direct_input_kv3",
            model_ir,
        )
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            output_keras_v3=True,
        )
        keras_path = os.path.join(
            output_dir,
            "add_tflite_direct_input_kv3_float32_v3.keras",
        )
        assert os.path.exists(keras_path)
        loaded = tf_keras.models.load_model(
            keras_path,
            compile=False,
            safe_mode=False,
        )
        assert loaded is not None
        outputs = loaded(
            [
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            ]
        )
        np.testing.assert_allclose(
            np.asarray(outputs),
            np.asarray([[5.0, 7.0, 9.0]], dtype=np.float32),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_tflite_direct_input_output_tfv1_pb_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(
            tmpdir,
            "add_tflite_direct_input_pb",
            model_ir,
        )
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            output_tfv1_pb=True,
        )
        assert os.path.exists(
            os.path.join(output_dir, "add_tflite_direct_input_pb_float32.pb")
        )


def test_tflite_direct_input_disable_model_save_leaves_no_artifacts() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(
            tmpdir,
            "add_tflite_direct_input_no_save",
            model_ir,
        )
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            disable_model_save=True,
        )
        _assert_dir_empty_or_missing(output_dir)


def test_tflite_direct_input_interrupt_input_names_model_ir_crop_smoke() -> None:
    model_ir = _make_add_relu_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(
            tmpdir,
            "add_relu_tflite_direct_input_inimc",
            model_ir,
        )
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            input_names_to_interrupt_model_conversion=["sum"],
        )
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))
        input_names, outputs = _run_saved_model_with_inputs(
            output_dir,
            {"sum": np.asarray([[-1.0, 2.0, -3.0]], dtype=np.float32)},
        )
        assert input_names == ["sum"]
        np.testing.assert_allclose(
            outputs["z"],
            np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_tflite_direct_input_interrupt_output_names_model_ir_crop_smoke() -> None:
    model_ir = _make_add_relu_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(
            tmpdir,
            "add_relu_tflite_direct_input_onimc",
            model_ir,
        )
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            output_names_to_interrupt_model_conversion=["sum"],
        )
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))
        input_names, outputs = _run_saved_model_with_inputs(
            output_dir,
            {
                "x": np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32),
                "y": np.asarray([[4.0, 5.0, -6.0]], dtype=np.float32),
            },
        )
        assert set(input_names) == {"x", "y"}
        np.testing.assert_allclose(
            outputs["sum"],
            np.asarray([[5.0, 3.0, -3.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_tflite_direct_input_saved_model_with_cotof_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_cotof", model_ir)
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            check_onnx_tf_outputs_elementwise_close_full=True,
        )
        report_path = os.path.join(
            output_dir,
            "add_tflite_direct_input_cotof_saved_model_validation_report.json",
        )
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["inference"]["status"] == "passed"
        assert report["comparison"]["status"] == "passed"
        assert report["comparison"]["pass"] is True
        assert report["overall_pass"] is True


def test_tflite_direct_input_split_outputs_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_split", model_ir)
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            enable_auto_split_model=True,
            auto_split_max_size="256MB",
        )
        assert os.path.exists(os.path.join(output_dir, "add_tflite_direct_input_split.tflite"))
        assert os.path.exists(os.path.join(output_dir, "add_tflite_direct_input_split_split_plan.json"))
        manifest_path = os.path.join(output_dir, "add_tflite_direct_input_split_split_manifest.json")
        assert os.path.exists(manifest_path)
        assert os.path.exists(os.path.join(output_dir, "add_tflite_direct_input_split_0001.tflite"))
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["base_model"] == "add_tflite_direct_input_split.tflite"
        assert len(manifest["partitions"]) == 1


def test_tflite_direct_input_split_saved_model_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_split_sm", model_ir)
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            enable_auto_split_model=True,
            auto_split_max_size="256MB",
            flatbuffer_direct_output_saved_model=True,
        )
        assert os.path.exists(os.path.join(output_dir, "add_tflite_direct_input_split_sm.tflite"))
        assert os.path.exists(os.path.join(output_dir, "saved_model.pb"))
        manifest_path = os.path.join(output_dir, "add_tflite_direct_input_split_sm_split_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert len(manifest["partitions"]) == 1
        saved_model_dir = manifest["partitions"][0]["saved_model_dir"]
        assert os.path.exists(os.path.join(output_dir, saved_model_dir, "saved_model.pb"))


def test_tflite_direct_input_split_saved_model_cotof_smoke() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_split_cotof", model_ir)
        output_dir = os.path.join(tmpdir, "sm_out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            enable_auto_split_model=True,
            auto_split_max_size="256MB",
            flatbuffer_direct_output_saved_model=True,
            check_onnx_tf_outputs_elementwise_close_full=True,
        )
        report_path = os.path.join(
            output_dir,
            "add_tflite_direct_input_split_cotof_saved_model_validation_report.json",
        )
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["source_label"] == "tflite_direct_input"
        assert report["comparison"]["status"] == "passed"
        assert report["comparison"]["pass"] is True
        assert report["overall_pass"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        (
            {
                "tflite_backend": "tf_converter",
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "output_integer_quantized_tflite": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "check_onnx_tf_outputs_elementwise_close": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "disable_suppression_flextranspose": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "disable_suppression_flexstridedslice": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "optimization_for_gpu_delegate": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "replace_argmax_to_reducemax_and_indices_is_int64": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "replace_argmax_to_reducemax_and_indices_is_float32": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "replace_argmax_to_fused_argmax_and_indices_is_int64": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "replace_argmax_to_fused_argmax_and_indices_is_float32": True,
            }
        ),
    ],
)
def test_tflite_direct_input_validation(kwargs: dict) -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_ng", model_ir)
        with pytest.raises(SystemExit):
            onnx2tf.convert(
                input_tflite_file_path=tflite_path,
                output_folder_path=tmpdir,
                verbosity="error",
                **kwargs,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "disable_model_save": True,
                "output_h5": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "disable_model_save": True,
                "output_keras_v3": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "disable_model_save": True,
                "output_tfv1_pb": True,
            }
        ),
        (
            {
                "tflite_backend": "flatbuffer_direct",
                "enable_auto_split_model": True,
                "output_h5": True,
            }
        ),
    ],
)
def test_tflite_direct_input_new_conflict_validation(kwargs: dict) -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_conflict", model_ir)
        with pytest.raises(SystemExit):
            onnx2tf.convert(
                input_tflite_file_path=tflite_path,
                output_folder_path=tmpdir,
                verbosity="error",
                **kwargs,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"disable_model_save": True, "output_h5": True},
        {"disable_model_save": True, "output_keras_v3": True},
        {"disable_model_save": True, "output_tfv1_pb": True},
        {"enable_auto_split_model": True, "output_h5": True},
    ],
)
def test_flatbuffer_direct_new_conflict_validation(kwargs: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_conflict", _make_add_model())
        with pytest.raises(SystemExit):
            onnx2tf.convert(
                input_onnx_file_path=model_path,
                output_folder_path=tmpdir,
                verbosity="error",
                disable_strict_mode=True,
                tflite_backend="flatbuffer_direct",
                **kwargs,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_names_to_interrupt_model_conversion": ["missing_tensor"]},
        {"output_names_to_interrupt_model_conversion": ["missing_tensor"]},
        {
            "input_names_to_interrupt_model_conversion": ["y"],
            "output_names_to_interrupt_model_conversion": ["sum"],
        },
    ],
)
def test_flatbuffer_direct_interrupt_validation(kwargs: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_relu_interrupt_ng", _make_add_relu_model())
        with pytest.raises(SystemExit):
            onnx2tf.convert(
                input_onnx_file_path=model_path,
                output_folder_path=tmpdir,
                verbosity="error",
                disable_strict_mode=True,
                tflite_backend="flatbuffer_direct",
                **kwargs,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_names_to_interrupt_model_conversion": ["missing_tensor"]},
        {"output_names_to_interrupt_model_conversion": ["missing_tensor"]},
        {
            "input_names_to_interrupt_model_conversion": ["y"],
            "output_names_to_interrupt_model_conversion": ["sum"],
        },
    ],
)
def test_tflite_direct_input_interrupt_validation(kwargs: dict) -> None:
    model_ir = _make_add_relu_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = _write_model_ir_as_tflite(
            tmpdir,
            "add_relu_tflite_direct_interrupt_ng",
            model_ir,
        )
        with pytest.raises(SystemExit):
            onnx2tf.convert(
                input_tflite_file_path=tflite_path,
                output_folder_path=tmpdir,
                verbosity="error",
                tflite_backend="flatbuffer_direct",
                **kwargs,
            )


def test_tflite_direct_input_rejects_mixed_onnx_and_tflite_input() -> None:
    model_ir = _make_add_model_ir()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add", _make_add_model())
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_direct_input_mix_ng", model_ir)
        with pytest.raises(SystemExit):
            onnx2tf.convert(
                input_onnx_file_path=model_path,
                input_tflite_file_path=tflite_path,
                output_folder_path=tmpdir,
                verbosity="error",
            )


def test_flatbuffer_direct_output_saved_model_cotof_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add", _make_add_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_saved_model=True,
            check_onnx_tf_outputs_elementwise_close_full=True,
        )
        assert os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
        assert os.path.exists(os.path.join(tmpdir, "add_float32.tflite"))
        report_path = os.path.join(
            tmpdir,
            "add_saved_model_validation_report.json",
        )
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["inference"]["status"] == "passed"
        assert report["comparison"]["status"] == "passed"
        assert report["comparison"]["pass"] is True
        assert report["overall_pass"] is True


def test_flatbuffer_direct_output_pytorch_cotof_generates_comparison_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_pytorch_cotof", _make_add_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_pytorch=True,
            check_onnx_tf_outputs_elementwise_close_full=True,
        )
        tflite_report_path = os.path.join(
            tmpdir,
            "add_pytorch_cotof_accuracy_report.json",
        )
        pytorch_report_path = os.path.join(
            tmpdir,
            "add_pytorch_cotof_pytorch_accuracy_report.json",
        )
        comparison_report_path = os.path.join(
            tmpdir,
            "add_pytorch_cotof_accuracy_comparison_report.json",
        )
        assert os.path.exists(tflite_report_path)
        assert os.path.exists(pytorch_report_path)
        assert os.path.exists(comparison_report_path)
        with open(comparison_report_path, "r", encoding="utf-8") as f:
            comparison_report = json.load(f)
        assert comparison_report["inputs_source"] == "seeded_random"
        assert comparison_report["onnx_tflite"] is not None
        assert comparison_report["onnx_pytorch"] is not None
        assert comparison_report["onnx_tflite"]["report_path"] == tflite_report_path
        assert comparison_report["onnx_pytorch"]["report_path"] == pytorch_report_path


def test_tflite_direct_input_pytorch_cotof_generates_comparison_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_ir = _make_add_model_ir()
        tflite_path = _write_model_ir_as_tflite(tmpdir, "add_tflite_input_pytorch_cotof", model_ir)
        output_dir = os.path.join(tmpdir, "out")
        onnx2tf.convert(
            input_tflite_file_path=tflite_path,
            output_folder_path=output_dir,
            verbosity="error",
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_pytorch=True,
            check_onnx_tf_outputs_elementwise_close_full=True,
            disable_model_save=True,
        )
        pytorch_report_path = os.path.join(
            output_dir,
            "add_tflite_input_pytorch_cotof_pytorch_accuracy_report.json",
        )
        comparison_report_path = os.path.join(
            output_dir,
            "add_tflite_input_pytorch_cotof_accuracy_comparison_report.json",
        )
        assert os.path.exists(pytorch_report_path)
        assert os.path.exists(comparison_report_path)
        with open(comparison_report_path, "r", encoding="utf-8") as f:
            comparison_report = json.load(f)
        assert comparison_report["inputs_source"] == "seeded_random"
        assert comparison_report["reference_backend"] == "tflite"
        assert comparison_report["onnx_tflite"] is None
        assert comparison_report["onnx_pytorch"] is None
        assert comparison_report["tflite_pytorch"] is not None
        assert comparison_report["tflite_pytorch"]["report_path"] == pytorch_report_path
        assert comparison_report["tflite_pytorch"]["evaluation_pass"] is True
        assert "overall_metrics" in comparison_report["tflite_pytorch"]


def test_flatbuffer_direct_output_saved_model_split_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_split_sm", _make_add_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_saved_model=True,
            enable_auto_split_model=True,
            auto_split_max_size="256MB",
        )
        assert not os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
        manifest_path = os.path.join(tmpdir, "add_split_sm_split_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert len(manifest["partitions"]) == 1
        saved_model_dir = manifest["partitions"][0]["saved_model_dir"]
        assert os.path.exists(os.path.join(tmpdir, saved_model_dir, "saved_model.pb"))
        assert os.path.exists(os.path.join(tmpdir, "add_split_sm_float32.tflite"))


def test_flatbuffer_direct_output_saved_model_split_cotof_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add_split_sm_cotof", _make_add_model())
        onnx2tf.convert(
            input_onnx_file_path=model_path,
            output_folder_path=tmpdir,
            verbosity="error",
            disable_strict_mode=True,
            tflite_backend="flatbuffer_direct",
            flatbuffer_direct_output_saved_model=True,
            enable_auto_split_model=True,
            auto_split_max_size="256MB",
            check_onnx_tf_outputs_elementwise_close_full=True,
        )
        report_path = os.path.join(
            tmpdir,
            "add_split_sm_cotof_saved_model_validation_report.json",
        )
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["mode"] == "split_saved_model"
        assert report["comparison"]["status"] == "passed"
        assert report["comparison"]["pass"] is True
        assert report["overall_pass"] is True


def test_flatbuffer_direct_output_saved_model_cotof_runs_saved_model_inference_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _save_model(tmpdir, "add", _make_add_model())

        def _raise_saved_model_load(*args, **kwargs):
            raise RuntimeError("saved model load failure for test")

        monkeypatch.setattr(tf.saved_model, "load", _raise_saved_model_load)
        with pytest.raises(RuntimeError, match="SavedModel inference check failed"):
            onnx2tf.convert(
                input_onnx_file_path=model_path,
                output_folder_path=tmpdir,
                verbosity="error",
                disable_strict_mode=True,
                tflite_backend="flatbuffer_direct",
                flatbuffer_direct_output_saved_model=True,
                check_onnx_tf_outputs_elementwise_close_full=True,
            )
        report_path = os.path.join(
            tmpdir,
            "add_saved_model_validation_report.json",
        )
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["inference"]["status"] == "failed"
        assert report["comparison"]["status"] == "skipped"
        assert report["overall_pass"] is False


def test_saved_model_exporter_kernel_registry_completeness() -> None:
    known_ops = get_known_model_ir_op_types() - {"CUSTOM", "MODEL"}
    supported_ops = get_supported_kernel_op_types()
    assert known_ops.issubset(supported_ops)


def test_saved_model_exporter_fail_fast_for_unknown_op_type() -> None:
    model_ir = ModelIR(name="unsupported_op_model")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.operators.append(
        OperatorIR(
            op_type="UNKNOWN_OP_TYPE_FOR_TEST",
            inputs=["x"],
            outputs=["y"],
            options={},
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ModelIRSavedModelExportError, match="unsupported_op_types"):
            export_saved_model_from_model_ir(
                model_ir=model_ir,
                output_folder_path=tmpdir,
            )
