from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from typing import Dict, Any

from onnx2tf.utils.common_functions import get_tflite_schema_fbs_url


def _run_cmd(cmd, cwd=None) -> None:
    subprocess.check_output(
        cmd,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )


@lru_cache(maxsize=16)
def load_schema_module(output_folder_path: str) -> Dict[str, Any]:
    os.makedirs(output_folder_path, exist_ok=True)

    _run_cmd(["flatc", "--version"])

    schema_fbs_path = os.path.join(output_folder_path, "schema.fbs")
    schema_py_path = os.path.join(output_folder_path, "schema_generated.py")

    if not os.path.isfile(schema_fbs_path):
        _run_cmd(
            [
                "curl",
                "-fL",
                get_tflite_schema_fbs_url(),
                "-o",
                schema_fbs_path,
            ]
        )

    if not os.path.isfile(schema_py_path):
        _run_cmd(
            [
                "flatc",
                "-t",
                "--python",
                "--gen-object-api",
                "--gen-onefile",
                "schema.fbs",
            ],
            cwd=output_folder_path,
        )

    schema_tflite: Dict[str, Any] = {}
    with open(schema_py_path, "r", encoding="utf-8") as f:
        exec(f.read(), schema_tflite)
    return schema_tflite
