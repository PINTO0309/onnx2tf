from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError
from onnx2tf.tflite_builder.pytorch_runtime_wrapper_exporter import (
    _export_runtime_wrapper_package_from_model_ir,
)


def _runtime_model_ir() -> ModelIR:
    model_ir = ModelIR(name="runtime_wrapper")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR(name="x", dtype="FLOAT32", shape=[1, 2]),
        "weight-value": TensorIR(
            name="weight-value",
            dtype="FLOAT32",
            shape=[2],
            data=np.asarray([1.0, 2.0], dtype=np.float32),
        ),
        "y": TensorIR(name="y", dtype="FLOAT32", shape=[1, 2]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="ADD",
            inputs=["x", "weight-value"],
            outputs=["y"],
        )
    ]
    return model_ir


def test_runtime_wrapper_export_writes_shared_package_metadata_and_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_state: dict[str, object] = {}

    def _save(state_dict: dict[str, object], path: str) -> None:
        saved_state.update(state_dict)
        Path(path).write_bytes(b"fake-state")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(as_tensor=lambda value: np.asarray(value), save=_save),
    )

    result = _export_runtime_wrapper_package_from_model_ir(
        model_ir=_runtime_model_ir(),
        output_folder_path=str(tmp_path),
    )

    assert result == str(tmp_path)
    assert (tmp_path / "__init__.py").is_file()
    assert (tmp_path / "runtime.py").is_file()
    assert (tmp_path / "model.py").is_file()
    assert (tmp_path / "state_dict.pth").read_bytes() == b"fake-state"
    assert set(saved_state) == {"weight_value"}
    np.testing.assert_array_equal(saved_state["weight_value"], [1.0, 2.0])
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "runtime_wrapper"
    assert metadata["tensor_storage_names"] == {"weight-value": "weight_value"}
    assert metadata["inputs"] == ["x"]
    assert metadata["outputs"] == ["y"]


def test_runtime_wrapper_export_rejects_unsupported_model_before_writing(
    tmp_path,
) -> None:
    model_ir = _runtime_model_ir()
    model_ir.operators[0].op_type = "UNKNOWN_RUNTIME_OP"
    output_path = tmp_path / "rejected"

    with pytest.raises(ModelIRPyTorchExportError, match="does not support"):
        _export_runtime_wrapper_package_from_model_ir(
            model_ir=model_ir,
            output_folder_path=str(output_path),
        )

    assert not output_path.exists()
