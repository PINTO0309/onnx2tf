import json
import os
import tempfile

import flatbuffers
import numpy as np
import onnx
import onnx2tf
import pytest
import tensorflow as tf
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
    module = tf.saved_model.load(saved_model_path)
    fn = module.signatures["serving_default"]
    outputs = fn(**{input_name: tf.constant(input_value)})
    return np.asarray(outputs[output_name].numpy())


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


def test_flatbuffer_direct_cotof_without_fdosm_skips_saved_model_check() -> None:
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
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["inference"]["status"] == "skipped"
        assert report["inference"]["reason"] == "saved_model_unavailable"


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
        module = tf.saved_model.load(saved_model_path)
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
                "disable_model_save": True,
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
