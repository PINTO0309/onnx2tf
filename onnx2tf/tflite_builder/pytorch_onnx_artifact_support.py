from __future__ import annotations

import json
import re
from pathlib import Path

import onnx

from onnx2tf.tflite_builder.pytorch_onnx_layout_passes import (
    _onnx_fold_pad_layout_bridges_in_place,
    _onnx_fold_relu_layout_bridges_in_place,
    _onnx_fold_residual_add_layout_bridges_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_optimizer import (
    _optimize_dynamo_exported_onnx_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _clear_onnx_graph_and_node_metadata_in_place,
    _onnx_repair_inferred_shapes_in_place,
)


def _onnx_model_uses_external_data(model: onnx.ModelProto) -> bool:
    return any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
    )


def _inspect_onnx_uses_external_data(onnx_path: Path) -> bool:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return _onnx_model_uses_external_data(model)


def _restore_missing_onnx_output_shapes_from_package_metadata(
    model: onnx.ModelProto,
    *,
    package_dir: Path,
) -> None:
    metadata_path = package_dir / "metadata.json"
    if not metadata_path.exists():
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return
    tensor_meta_map = metadata.get("tensors", {})
    if not isinstance(tensor_meta_map, dict):
        return
    for output in model.graph.output:
        tensor_type = getattr(output.type, "tensor_type", None)
        if tensor_type is None:
            continue
        shape = getattr(tensor_type, "shape", None)
        if shape is None or len(list(shape.dim)) != 0:
            continue
        tensor_meta = tensor_meta_map.get(str(output.name), {})
        if not isinstance(tensor_meta, dict):
            continue
        shape_values = tensor_meta.get("shape_signature", tensor_meta.get("shape", []))
        if not isinstance(shape_values, list):
            shape_values = tensor_meta.get("shape", [])
        if not isinstance(shape_values, list) or len(shape_values) == 0:
            continue
        for dim_index, raw_dim_value in enumerate(shape_values):
            output_dim = shape.dim.add()
            try:
                dim_value = int(raw_dim_value)
            except Exception:
                output_dim.dim_param = str(raw_dim_value)
                continue
            if dim_value > 0:
                output_dim.dim_value = dim_value
            else:
                sanitized_output_name = re.sub(
                    r"[^0-9A-Za-z_]", "_", str(output.name)
                ).strip("_")
                if sanitized_output_name == "":
                    sanitized_output_name = "output"
                output_dim.dim_param = f"{sanitized_output_name}_dim_{dim_index}"


def _sanitize_dynamo_exported_onnx_metadata(onnx_path: Path) -> None:
    external_data_sidecar_path = onnx_path.with_name(f"{onnx_path.name}.data")
    original_uses_external_data = _inspect_onnx_uses_external_data(onnx_path)
    model = onnx.load(str(onnx_path))
    _onnx_fold_relu_layout_bridges_in_place(model.graph)
    _onnx_fold_pad_layout_bridges_in_place(model.graph)
    _onnx_fold_residual_add_layout_bridges_in_place(model.graph)
    _onnx_repair_inferred_shapes_in_place(model)
    _onnx_fold_residual_add_layout_bridges_in_place(model.graph)
    _onnx_repair_inferred_shapes_in_place(model)
    _optimize_dynamo_exported_onnx_in_place(model)
    import onnx2tf.gs as gs

    model = gs.export_onnx(gs.import_onnx(model).cleanup().toposort())
    _onnx_repair_inferred_shapes_in_place(model)
    _restore_missing_onnx_output_shapes_from_package_metadata(
        model, package_dir=onnx_path.parent
    )
    del model.metadata_props[:]
    _clear_onnx_graph_and_node_metadata_in_place(model.graph)
    onnx.checker.check_model(model)
    if original_uses_external_data:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_sidecar_path.name,
            size_threshold=0,
        )
    else:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=False,
        )
    if (
        not _inspect_onnx_uses_external_data(onnx_path)
        and external_data_sidecar_path.exists()
    ):
        external_data_sidecar_path.unlink()
