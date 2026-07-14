from __future__ import annotations

from pathlib import Path
from typing import Optional


def _write_generated_package_common_files(
    output_folder_path: str,
    *,
    runtime_source: Optional[str] = None,
) -> None:
    package_dir = Path(output_folder_path)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "import logging\n"
        "logging.getLogger('torch.onnx._internal.exporter._registration').setLevel(logging.ERROR)\n"
        "from .model import Model, load_model\n",
        encoding="utf-8",
    )
    if runtime_source is None:
        runtime_source = (
            "# pyright: reportArgumentType=false, reportCallIssue=false\n"
            "from onnx2tf.tflite_builder.pytorch_package_runtime import load_generated_model_package\n"
        )
    (package_dir / "runtime.py").write_text(
        runtime_source,
        encoding="utf-8",
    )


def _patch_generated_runtime_pool2d_channel_last_recovery(package_dir: Path) -> None:
    runtime_path = package_dir / "runtime.py"
    if not runtime_path.exists():
        return
    lines = runtime_path.read_text(encoding="utf-8").splitlines()
    if any("int(actual_shape[-1]) == int(target[1])" in line for line in lines):
        return
    in_apply_pool2d = False
    insert_at: Optional[int] = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def _apply_pool2d("):
            in_apply_pool2d = True
            continue
        if not in_apply_pool2d:
            continue
        if stripped.startswith("def "):
            break
        if stripped.startswith(
            "if resize_as_channel_last and x.ndim == 4 and has_target_shape and len(target) == 4:"
        ):
            insert_at = index
            break
    if insert_at is None:
        return
    insertion = [
        "    if not resize_as_channel_last and x.ndim == 4 and has_target_shape and len(target) == 4:",
        "        has_actual_shape, actual_shape = _tensor_static_shape_list(x)",
        "        if has_actual_shape and int(actual_shape[-1]) == int(target[1]) and int(actual_shape[1]) != int(target[1]):",
        "            resize_as_channel_last = True",
    ]
    lines[insert_at:insert_at] = insertion
    runtime_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_wrapper_model_file(output_folder_path: str) -> None:
    package_dir = Path(output_folder_path)
    (package_dir / "model.py").write_text(
        "# pyright: reportArgumentType=false, reportCallIssue=false\n"
        "from __future__ import annotations\n\n"
        "from typing import Any, Callable, cast\n\n"
        "import logging\n"
        "from pathlib import Path\n\n"
        "import torch\n\n"
        "logging.getLogger('torch.onnx._internal.exporter._registration').setLevel(logging.ERROR)\n\n"
        "from .runtime import load_generated_model_package\n\n"
        "PACKAGE_DIR = Path(__file__).resolve().parent\n\n"
        "class Model(torch.nn.Module):\n"
        "    def __init__(self, device: str | None = None, eval_mode: bool = True):\n"
        "        super().__init__()\n"
        "        self._model: Any = load_generated_model_package(\n"
        "            package_dir=str(PACKAGE_DIR),\n"
        "            device=device,\n"
        "            eval_mode=eval_mode,\n"
        "        )\n\n"
        "    def forward(self, *args: Any, **kwargs: Any) -> Any:\n"
        "        return self._model(*args, **kwargs)\n\n"
        "    def forward_named(self, *args: Any, **kwargs: Any) -> Any:\n"
        "        forward_named = getattr(self._model, 'forward_named', None)\n"
        "        if callable(forward_named):\n"
        "            return cast(Callable[..., Any], forward_named)(*args, **kwargs)\n"
        "        return self.forward(*args, **kwargs)\n\n"
        "def load_model(device: str | None = None, eval_mode: bool = True) -> Model:\n"
        "    return Model(device=device, eval_mode=eval_mode)\n",
        encoding="utf-8",
    )


def _build_native_runtime_source(helper_source: str) -> str:
    runtime_source = (
        "# pyright: reportArgumentType=false, reportCallIssue=false\n"
        "from pathlib import Path\n"
        "import re\n"
        "from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple\n\n"
        "import numpy as np\n"
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        f"{helper_source}"
        "def _resolve_model_attribute(model: torch.nn.Module, attr_path: str) -> Any:\n"
        "    value: Any = model\n"
        "    for part in str(attr_path).split('.'):\n"
        "        value = getattr(value, part)\n"
        "    return value\n\n"
        "def resolve_model_tensor(model: torch.nn.Module, attr_name: str) -> torch.Tensor:\n"
        "    value = _resolve_model_attribute(model, attr_name)\n"
        "    if not isinstance(value, torch.Tensor):\n"
        "        raise RuntimeError(f'Generated model attribute is not a tensor: {attr_name}')\n"
        "    return value\n\n"
        "def load_generated_weights(\n"
        "    *,\n"
        "    model: torch.nn.Module,\n"
        "    package_dir: Path,\n"
        "    device: Optional[str],\n"
        ") -> None:\n"
        "    raw_state_dict = torch.load(package_dir / 'state_dict.pth', map_location=device or 'cpu')\n"
        "    model.load_state_dict(raw_state_dict, strict=True)\n"
        "    if device is not None:\n"
        "        model.to(device)\n"
    )
    runtime_source = runtime_source.replace(
        "Optional[Sequence[int]]", "Optional[List[int]]"
    )
    runtime_source = runtime_source.replace("Sequence[int]", "List[int]")
    runtime_source = runtime_source.replace("Sequence[str]", "List[str]")
    runtime_source = runtime_source.replace(
        "Sequence[torch.Tensor]", "List[torch.Tensor]"
    )
    runtime_source = runtime_source.replace("Sequence[Any]", "List[Any]")
    return runtime_source
