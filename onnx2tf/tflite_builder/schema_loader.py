from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Any

from onnx2tf.utils.common_functions import ensure_tflite_schema_artifacts


@lru_cache(maxsize=16)
def load_schema_module(output_folder_path: str) -> Dict[str, Any]:
    os.makedirs(output_folder_path, exist_ok=True)

    _, schema_py_path = ensure_tflite_schema_artifacts(
        output_folder_path=output_folder_path,
        force_regenerate_schema_py=False,
    )

    schema_tflite: Dict[str, Any] = {}
    with open(schema_py_path, "r", encoding="utf-8") as f:
        exec(f.read(), schema_tflite)
    return schema_tflite
