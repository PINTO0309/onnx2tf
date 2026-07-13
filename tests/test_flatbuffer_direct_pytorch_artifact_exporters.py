from __future__ import annotations

import ast
import hashlib
import json
import sys
import types

from onnx2tf.tflite_builder.pytorch_artifact_exporters import (
    _export_dynamo_onnx_from_generated_package,
    _export_exported_program_from_generated_package,
    export_torchscript_from_generated_package,
)
from onnx2tf.tflite_builder.pytorch_export_support import (
    _generated_package_non_native_skip_reason,
    _metadata_has_dynamic_public_inputs,
)
from onnx2tf.tflite_builder.pytorch_exported_program_child import (
    _EXPORTED_PROGRAM_CHILD_SCRIPT,
)


def test_exported_program_child_payload_is_fixed_and_parseable() -> None:
    assert len(_EXPORTED_PROGRAM_CHILD_SCRIPT.encode("utf-8")) == 71054
    assert hashlib.sha256(
        _EXPORTED_PROGRAM_CHILD_SCRIPT.encode("utf-8")
    ).hexdigest() == (
        "548c123d658c61780a134e34dbc02939f07d1db7e6bccc81db08fddf6cf77d5e"
    )
    ast.parse(_EXPORTED_PROGRAM_CHILD_SCRIPT)


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
        strip_stack_traces_from_exported_program_archive_fn=unexpected_hook,
        fold_inverse_permute_round_trips_in_exported_program_archive_fn=(
            unexpected_hook
        ),
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
