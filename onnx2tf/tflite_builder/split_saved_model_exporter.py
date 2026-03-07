from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from onnx2tf.tflite_builder.ir import (
    optimize_redundant_transpose_operators,
    prune_identity_cast_operators,
)
from onnx2tf.tflite_builder.saved_model_exporter import (
    export_saved_model_from_model_ir,
)
from onnx2tf.tflite_builder.split_planner import build_partition_model_ir


def export_split_saved_models(
    *,
    model_ir: Any,
    split_manifest_path: str,
    output_folder_path: str,
    output_file_name: str,
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

    saved_model_dirs: List[str] = []
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
        saved_model_dir_name = f"{output_file_name}_saved_model_{partition_id:04d}"
        export_saved_model_from_model_ir(
            model_ir=part_model_ir,
            output_folder_path=os.path.join(output_folder_path, saved_model_dir_name),
        )
        partition["saved_model_dir"] = saved_model_dir_name
        saved_model_dirs.append(saved_model_dir_name)

    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "split_saved_model_dirs": saved_model_dirs,
        "split_saved_model_count": len(saved_model_dirs),
    }
