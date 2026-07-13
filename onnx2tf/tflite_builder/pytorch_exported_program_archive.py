from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def _strip_stack_traces_from_exported_program_archive(
    exported_program_path: Path,
) -> None:
    archive_path = Path(exported_program_path)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"ExportedProgram archive not found. path={archive_path}"
        )
    with tempfile.NamedTemporaryFile(
        prefix="onnx2tf_exported_program_strip_",
        suffix=".pt2",
        delete=False,
        dir=str(archive_path.parent),
    ) as tmp_file:
        temp_archive_path = Path(tmp_file.name)
    try:
        removed_count = 0
        with (
            zipfile.ZipFile(str(archive_path), "r") as source_archive,
            zipfile.ZipFile(
                str(temp_archive_path),
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as stripped_archive,
        ):
            for info in source_archive.infolist():
                payload = source_archive.read(info.filename)
                if info.filename.endswith("models/model.json"):
                    model_json = json.loads(payload)

                    def _strip_stack_trace_fields(value: Any) -> None:
                        nonlocal removed_count
                        if isinstance(value, dict):
                            if "stack_trace" in value:
                                del value["stack_trace"]
                                removed_count += 1
                            for child in value.values():
                                _strip_stack_trace_fields(child)
                            return
                        if isinstance(value, list):
                            for child in value:
                                _strip_stack_trace_fields(child)

                    _strip_stack_trace_fields(model_json)
                    payload = json.dumps(model_json, separators=(",", ":")).encode(
                        "utf-8"
                    )
                stripped_archive.writestr(info, payload)
        if removed_count == 0:
            temp_archive_path.unlink(missing_ok=True)
            return
        temp_archive_path.replace(archive_path)
    except Exception:
        temp_archive_path.unlink(missing_ok=True)
        raise
