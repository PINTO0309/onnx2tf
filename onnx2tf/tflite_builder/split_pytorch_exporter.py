from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.ir import (
    optimize_redundant_transpose_operators,
    prune_identity_cast_operators,
)
from onnx2tf.tflite_builder.pytorch_exporter import (
    export_pytorch_package_from_model_ir,
    export_torchscript_from_generated_package,
)
from onnx2tf.tflite_builder.split_planner import build_partition_model_ir


def export_split_pytorch_packages(
    *,
    model_ir: Any,
    split_manifest_path: str,
    output_folder_path: str,
    output_file_name: str,
    output_torchscript_from_model_ir: bool = False,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(
            f"Split manifest does not exist. path={split_manifest_path}"
        )

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    partitions = list(manifest.get("partitions", []))
    if len(partitions) == 0:
        raise ValueError("Split manifest contains no partition entries.")

    package_dirs: List[str] = []
    torchscript_paths: List[str] = []
    for partition in partitions:
        partition_id = int(partition["partition_id"])
        part_model_ir = build_partition_model_ir(
            model_ir=model_ir,
            start_op_index=int(partition["start_op_index"]),
            end_op_index=int(partition["end_op_index"]),
            partition_id=partition_id,
        )
        prune_identity_cast_operators(
            part_model_ir,
            preserve_model_outputs=True,
        )
        optimize_redundant_transpose_operators(
            part_model_ir,
            preserve_model_outputs=True,
        )
        package_dir_name = f"{output_file_name}_pytorch_{partition_id:04d}"
        package_dir_path = export_pytorch_package_from_model_ir(
            model_ir=part_model_ir,
            output_folder_path=os.path.join(output_folder_path, package_dir_name),
        )
        if output_torchscript_from_model_ir:
            package_input_names = [str(v) for v in list(getattr(part_model_ir, "inputs", []))]
            filtered_custom_inputs = None
            if custom_input_op_name_np_data_path is not None:
                filtered_custom_inputs = [
                    param
                    for param in custom_input_op_name_np_data_path
                    if isinstance(param, (list, tuple))
                    and len(param) >= 2
                    and str(param[0]) in package_input_names
                ]
            torchscript_path = export_torchscript_from_generated_package(
                package_dir=str(package_dir_path),
                custom_input_op_name_np_data_path=filtered_custom_inputs,
                shape_hints=shape_hints,
                test_data_nhwc_path=test_data_nhwc_path,
            )
            partition["pytorch_torchscript_file_name"] = os.path.basename(torchscript_path)
            torchscript_paths.append(str(torchscript_path))
        partition["pytorch_package_dir"] = package_dir_name
        package_dirs.append(package_dir_name)

    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    outputs = {
        "split_pytorch_package_dirs": package_dirs,
        "split_pytorch_package_count": len(package_dirs),
    }
    if len(torchscript_paths) > 0:
        outputs["split_pytorch_torchscript_paths"] = list(torchscript_paths)
        outputs["split_pytorch_torchscript_count"] = int(len(torchscript_paths))
    return outputs
