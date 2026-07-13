from __future__ import annotations

import json
import sys
import types

from onnx2tf.tflite_builder.pytorch_artifact_exporters import (
    export_torchscript_from_generated_package,
)
from onnx2tf.tflite_builder.pytorch_export_support import (
    _generated_package_non_native_skip_reason,
    _metadata_has_dynamic_public_inputs,
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
