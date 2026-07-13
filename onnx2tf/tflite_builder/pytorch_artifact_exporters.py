from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional

from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)
from onnx2tf.tflite_builder.pytorch_exported_program_child import (
    _EXPORTED_PROGRAM_CHILD_SCRIPT,
)
from onnx2tf.tflite_builder.pytorch_export_support import (
    _build_pytorch_export_example_inputs,
    _generated_package_non_native_skip_reason,
    _generated_package_torch_export_skip_reason,
    _load_generated_package_export_metadata,
    _metadata_has_dynamic_public_inputs,
    _remove_generated_package_artifact_if_exists,
    _run_generated_package_export_child,
    _sanitize_torchscript_file_stem,
    _write_generated_package_export_metadata,
)
from onnx2tf.tflite_builder.pytorch_onnx_artifact_support import (
    _sanitize_dynamo_exported_onnx_metadata,
)


def export_torchscript_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    native_package_generation_timeout_sec: Optional[int] = 0,
    raise_on_failure: bool = True,
) -> Optional[str]:
    try:
        import torch as _torch

        del _torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "TorchScript export requires `torch` to be installed."
        ) from ex

    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="TorchScript export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
            extra_fields={
                "trace_mode": None,
            },
        )
        return None
    skip_reason = _generated_package_non_native_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "trace_mode": None,
                "skipped_reason": skip_reason,
            },
        )
        return None
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = (
            _build_pytorch_export_example_inputs(
                package_dir=package_dir,
                package_metadata=metadata,
                custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                shape_hints=shape_hints,
                test_data_nhwc_path=test_data_nhwc_path,
                export_label="TorchScript export",
            )
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
            extra_fields={
                "trace_mode": None,
            },
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    torchscript_file_name = f"{file_stem}_jit.pt"
    torchscript_path = package_path / torchscript_file_name
    child_script = """
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
torchscript_path = Path(sys.argv[3])
mode = str(sys.argv[4]).strip().lower()

module_name = (
    "_onnx2tf_generated_torchscript_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
with torch.no_grad():
    if mode == "trace":
        artifact = torch.jit.trace(model, example_inputs, check_trace=False)
    elif mode == "script":
        artifact = torch.jit.script(model)
    else:
        raise RuntimeError(f"Unsupported torchscript export mode: {mode}")
    torch.jit.save(artifact, str(torchscript_path))
print(json.dumps({"trace_mode": mode}))
"""
    trace_mode = ""
    last_error_message = ""
    timeout_sec = int(native_package_generation_timeout_sec or 0)
    for candidate_mode in ("trace", "script"):
        child_payload, last_error_message = _run_generated_package_export_child(
            example_inputs=example_inputs,
            child_script=child_script,
            package_path=package_path,
            artifact_path=torchscript_path,
            child_payload={},
            child_args=[candidate_mode],
            temp_prefix="onnx2tf_torchscript_",
            timeout_sec=timeout_sec,
        )
        if child_payload is not None:
            trace_mode = str(child_payload.get("trace_mode", candidate_mode))
            break
        if last_error_message != "":
            last_error_message = f"mode={candidate_mode} {last_error_message}"
    if trace_mode == "":
        _remove_generated_package_artifact_if_exists(torchscript_path)
        extra_fields = {
            "trace_mode": None,
        }
        if timeout_sec > 0 and "timed out after" in str(last_error_message):
            extra_fields.update(
                {
                    "timed_out": True,
                    "timeout_sec": int(timeout_sec),
                }
            )
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message,
            extra_fields=extra_fields,
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "TorchScript export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        return None
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="torchscript",
        file_name=str(torchscript_file_name),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
        extra_fields={
            "trace_mode": trace_mode,
        },
    )
    return str(torchscript_path)


def _export_dynamo_onnx_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    native_package_generation_timeout_sec: Optional[int] = 0,
    raise_on_failure: bool = True,
    temporarily_rewrite_generated_model_source_for_exported_program_fn: Callable[
        ..., Any
    ],
    reapply_post_export_final_model_repairs_fn: Callable[[Path], None],
) -> Optional[str]:
    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="Dynamo ONNX export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        return None
    skip_reason = _generated_package_torch_export_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "skipped_reason": skip_reason,
            },
        )
        return None
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = (
            _build_pytorch_export_example_inputs(
                package_dir=package_dir,
                package_metadata=metadata,
                custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                shape_hints=shape_hints,
                test_data_nhwc_path=test_data_nhwc_path,
                export_label="Dynamo ONNX export",
            )
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    dynamo_onnx_file_name = f"{file_stem}_dynamo.onnx"
    dynamo_onnx_path = package_path / dynamo_onnx_file_name
    timeout_sec = int(native_package_generation_timeout_sec or 0)
    child_script = """
import hashlib
import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
dynamo_onnx_path = Path(sys.argv[3])

module_name = (
    "_onnx2tf_generated_dynamo_onnx_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
input_names = [str(v) for v in list(payload.get("input_names", []))]
output_names = [str(v) for v in list(payload.get("output_names", []))]
logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(logging.ERROR)
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
with torch.no_grad():
    torch.onnx.export(
        model,
        example_inputs,
        str(dynamo_onnx_path),
        dynamo=True,
        input_names=input_names,
        output_names=output_names,
    )
print(json.dumps({"file_name": dynamo_onnx_path.name}))
    """
    with temporarily_rewrite_generated_model_source_for_exported_program_fn(
        package_path,
        model_ir=None,
    ):
        child_payload, last_error_message = _run_generated_package_export_child(
            example_inputs=example_inputs,
            child_script=child_script,
            package_path=package_path,
            artifact_path=dynamo_onnx_path,
            child_payload={
                "input_names": [str(v) for v in list(metadata.get("inputs", []))],
                "output_names": [str(v) for v in list(metadata.get("outputs", []))],
            },
            temp_prefix="onnx2tf_dynamo_onnx_",
            timeout_sec=timeout_sec,
        )
    if child_payload is None or not dynamo_onnx_path.exists():
        _remove_generated_package_artifact_if_exists(dynamo_onnx_path)
        extra_fields = None
        if timeout_sec > 0 and "timed out after" in str(last_error_message):
            extra_fields = {
                "timed_out": True,
                "timeout_sec": int(timeout_sec),
            }
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message
            or "dynamo=True ONNX export did not produce an artifact.",
            extra_fields=extra_fields,
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "Dynamo ONNX export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        reapply_post_export_final_model_repairs_fn(package_path)
        return None
    try:
        _sanitize_dynamo_exported_onnx_metadata(dynamo_onnx_path)
    except Exception as ex:
        _remove_generated_package_artifact_if_exists(dynamo_onnx_path)
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=str(ex),
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "Dynamo ONNX sanitize failed for the generated native PyTorch package. "
                f"package_dir={package_dir} artifact={dynamo_onnx_path} details={ex}"
            ) from ex
        reapply_post_export_final_model_repairs_fn(package_path)
        return None
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="dynamo_onnx",
        file_name=str(child_payload.get("file_name", dynamo_onnx_file_name)),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
    )
    reapply_post_export_final_model_repairs_fn(package_path)
    return str(dynamo_onnx_path)


def _export_exported_program_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    native_package_generation_timeout_sec: Optional[int] = 0,
    raise_on_failure: bool = True,
    temporarily_rewrite_generated_model_source_for_exported_program_fn: Callable[
        ..., Any
    ],
    reapply_post_export_final_model_repairs_fn: Callable[[Path], None],
    strip_stack_traces_from_exported_program_archive_fn: Callable[[Path], None],
    fold_inverse_permute_round_trips_in_exported_program_archive_fn: Callable[
        [Path], None
    ],
) -> Optional[str]:
    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="ExportedProgram export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        return None
    skip_reason = _generated_package_torch_export_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "skipped_reason": skip_reason,
            },
        )
        return None
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = (
            _build_pytorch_export_example_inputs(
                package_dir=package_dir,
                package_metadata=metadata,
                custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                shape_hints=shape_hints,
                test_data_nhwc_path=test_data_nhwc_path,
                export_label="ExportedProgram export",
            )
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    exported_program_file_name = f"{file_stem}_ep.pt2"
    exported_program_path = package_path / exported_program_file_name
    timeout_sec = int(native_package_generation_timeout_sec or 0)
    child_script = _EXPORTED_PROGRAM_CHILD_SCRIPT
    with temporarily_rewrite_generated_model_source_for_exported_program_fn(
        package_path,
        model_ir=None,
    ):
        child_payload, last_error_message = _run_generated_package_export_child(
            example_inputs=example_inputs,
            child_script=child_script,
            package_path=package_path,
            artifact_path=exported_program_path,
            child_payload={},
            temp_prefix="onnx2tf_exported_program_",
            timeout_sec=timeout_sec,
        )
    if child_payload is None or not exported_program_path.exists():
        _remove_generated_package_artifact_if_exists(exported_program_path)
        extra_fields = None
        if timeout_sec > 0 and "timed out after" in str(last_error_message):
            extra_fields = {
                "timed_out": True,
                "timeout_sec": int(timeout_sec),
            }
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message
            or "torch.export.save did not produce an artifact.",
            extra_fields=extra_fields,
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "ExportedProgram export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        reapply_post_export_final_model_repairs_fn(package_path)
        return None
    try:
        strip_stack_traces_from_exported_program_archive_fn(exported_program_path)
    except Exception as ex:
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "ExportedProgram archive cleanup failed for the generated native PyTorch package. "
                f"package_dir={package_dir} artifact={exported_program_path} details={ex}"
            ) from ex
        last_error_message = str(ex)
    try:
        fold_inverse_permute_round_trips_in_exported_program_archive_fn(
            exported_program_path
        )
    except Exception:
        pass
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="exported_program",
        file_name=str(child_payload.get("file_name", exported_program_file_name)),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
    )
    reapply_post_export_final_model_repairs_fn(package_path)
    return str(exported_program_path)
