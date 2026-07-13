from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
import zipfile

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_artifact_exporters import (
    _export_dynamo_onnx_from_generated_package,
    _export_exported_program_from_generated_package,
    export_torchscript_from_generated_package,
)
from onnx2tf.tflite_builder.pytorch_export_support import (
    _build_metadata_payload,
    _generated_package_non_native_skip_reason,
    _metadata_has_dynamic_public_inputs,
    _serializable_value,
)
from onnx2tf.tflite_builder.pytorch_exported_program_child import (
    _EXPORTED_PROGRAM_CHILD_SCRIPT,
)
from onnx2tf.tflite_builder.pytorch_exported_program_archive import (
    _fold_inverse_permute_round_trips_in_exported_program_archive,
    _strip_stack_traces_from_exported_program_archive,
)


def test_exported_program_child_payload_is_fixed_and_parseable() -> None:
    assert len(_EXPORTED_PROGRAM_CHILD_SCRIPT.encode("utf-8")) == 71054
    assert hashlib.sha256(
        _EXPORTED_PROGRAM_CHILD_SCRIPT.encode("utf-8")
    ).hexdigest() == (
        "548c123d658c61780a134e34dbc02939f07d1db7e6bccc81db08fddf6cf77d5e"
    )
    ast.parse(_EXPORTED_PROGRAM_CHILD_SCRIPT)


def test_exported_program_archive_cleanup_strips_only_stack_traces(
    tmp_path,
) -> None:
    archive_path = tmp_path / "model_ep.pt2"
    model_payload = {
        "graph": {
            "stack_trace": "root trace",
            "nodes": [
                {
                    "name": "node_0",
                    "metadata": {"stack_trace": "node trace", "keep": 7},
                }
            ],
        }
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/models/model.json", json.dumps(model_payload))
        archive.writestr("data/weights.bin", b"unchanged")

    _strip_stack_traces_from_exported_program_archive(archive_path)

    with zipfile.ZipFile(archive_path, "r") as archive:
        cleaned = json.loads(archive.read("data/models/model.json"))
        assert archive.read("data/weights.bin") == b"unchanged"
    assert cleaned == {
        "graph": {"nodes": [{"name": "node_0", "metadata": {"keep": 7}}]}
    }


def test_exported_program_archive_optimizer_rejects_missing_archive(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError, match="ExportedProgram archive not found"):
        _fold_inverse_permute_round_trips_in_exported_program_archive(
            tmp_path / "missing.pt2"
        )


def test_torchscript_export_records_non_native_skip_without_child(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "name": "runtime-wrapper",
                "execution_backend": "runtime_wrapper",
                "inputs": ["x"],
                "tensors": {
                    "x": {
                        "shape": [1, 3],
                        "shape_signature": [-1, 3],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = export_torchscript_from_generated_package(
        package_dir=str(tmp_path),
    )

    assert result is None
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    artifact_metadata = metadata["torchscript"]
    assert artifact_metadata["file_name"] is None
    assert artifact_metadata["dynamic_inputs_present"] is True
    assert artifact_metadata["trace_mode"] is None
    assert "non-native execution backend" in artifact_metadata["skipped_reason"]


def test_dynamo_export_records_non_native_skip_without_hooks(tmp_path) -> None:
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "name": "runtime-wrapper",
                "execution_backend": "runtime_wrapper",
                "inputs": ["x"],
                "tensors": {"x": {"shape": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    def unexpected_hook(*args, **kwargs):
        raise AssertionError(f"unexpected hook call: args={args} kwargs={kwargs}")

    result = _export_dynamo_onnx_from_generated_package(
        package_dir=str(tmp_path),
        temporarily_rewrite_generated_model_source_for_exported_program_fn=(
            unexpected_hook
        ),
        reapply_post_export_final_model_repairs_fn=unexpected_hook,
    )

    assert result is None
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    artifact_metadata = metadata["dynamo_onnx"]
    assert artifact_metadata["file_name"] is None
    assert artifact_metadata["dynamic_inputs_present"] is False
    assert "non-native execution backend" in artifact_metadata["skipped_reason"]


def test_exported_program_records_non_native_skip_without_hooks(
    tmp_path,
) -> None:
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "name": "runtime-wrapper",
                "execution_backend": "runtime_wrapper",
                "inputs": ["x"],
                "tensors": {"x": {"shape": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    def unexpected_hook(*args, **kwargs):
        raise AssertionError(f"unexpected hook call: args={args} kwargs={kwargs}")

    result = _export_exported_program_from_generated_package(
        package_dir=str(tmp_path),
        temporarily_rewrite_generated_model_source_for_exported_program_fn=(
            unexpected_hook
        ),
        reapply_post_export_final_model_repairs_fn=unexpected_hook,
    )

    assert result is None
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    artifact_metadata = metadata["exported_program"]
    assert artifact_metadata["file_name"] is None
    assert artifact_metadata["dynamic_inputs_present"] is False
    assert "non-native execution backend" in artifact_metadata["skipped_reason"]


def test_non_native_skip_policy_detects_legacy_runtime_wrapper(tmp_path) -> None:
    (tmp_path / "model.py").write_text(
        "load_generated_model_package()\n",
        encoding="utf-8",
    )

    reason = _generated_package_non_native_skip_reason(tmp_path)

    assert reason is not None
    assert "runtime_wrapper" in reason


def test_dynamic_public_input_policy_uses_shape_signature() -> None:
    assert _metadata_has_dynamic_public_inputs(
        {
            "inputs": ["x"],
            "tensors": {
                "x": {
                    "shape": [1, 3, 224, 224],
                    "shape_signature": [-1, 3, 224, 224],
                }
            },
        }
    )
    assert not _metadata_has_dynamic_public_inputs(
        {
            "inputs": ["x"],
            "tensors": {
                "x": {
                    "shape": [1, 3, 224, 224],
                    "shape_signature": [1, 3, 224, 224],
                }
            },
        }
    )


def test_serializable_value_normalizes_nested_numpy_values() -> None:
    assert _serializable_value(
        {
            "array": np.asarray([1, 2], dtype=np.int32),
            "scalar": np.float32(0.5),
            "tuple": (np.int64(3),),
        }
    ) == {
        "array": [1, 2],
        "scalar": 0.5,
        "tuple": [3],
    }


def test_metadata_payload_restores_public_boundary_contract() -> None:
    model_ir = ModelIR(name="metadata_boundary_contract", description="fixture")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 8, 8],
        shape_signature=[1, 3, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.tensors["const"] = TensorIR(
        name="const",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2.0], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 8, 8],
        shape_signature=[1, 4, 8, 8],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ADD",
            inputs=["x", "const"],
            outputs=["y"],
            options={"alpha": np.float32(1.0)},
            axis_semantics={"axis": np.int64(1)},
            version=2,
        )
    )
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [-1, 8, 8, 3],
        "y": [-1, 8, 8, 4],
    }
    model_ir.metadata["onnx_public_layout_map"] = {"x": "NHWC", "y": "NHWC"}

    payload = _build_metadata_payload(model_ir)

    assert payload["schema_version"] == 1
    assert payload["inputs"] == ["x"]
    assert payload["outputs"] == ["y"]
    assert payload["tensors"]["x"]["shape"] == [1, 8, 8, 3]
    assert payload["tensors"]["x"]["shape_signature"] == [-1, 8, 8, 3]
    assert payload["tensors"]["x"]["logical_layout"] == "NHWC"
    assert payload["tensors"]["const"]["has_data"] is True
    assert payload["current_public_layouts"] == {"x": "NCHW", "y": "NCHW"}
    assert payload["operators"] == [
        {
            "op_type": "ADD",
            "inputs": ["x", "const"],
            "outputs": ["y"],
            "options": {"alpha": 1.0},
            "axis_semantics": {"axis": 1},
            "version": 2,
        }
    ]
