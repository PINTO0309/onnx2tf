from __future__ import annotations

import ast

from onnx2tf.tflite_builder.pytorch_package_sources import (
    _build_native_runtime_source,
    _patch_generated_runtime_pool2d_channel_last_recovery,
    _write_generated_package_common_files,
    _write_wrapper_model_file,
)


def test_common_package_files_use_default_runtime_bridge(tmp_path) -> None:
    package_dir = tmp_path / "nested" / "generated"

    _write_generated_package_common_files(str(package_dir))

    assert (package_dir / "__init__.py").read_text(encoding="utf-8") == (
        "import logging\n"
        "logging.getLogger('torch.onnx._internal.exporter._registration').setLevel(logging.ERROR)\n"
        "from .model import Model, load_model\n"
    )
    assert (package_dir / "runtime.py").read_text(encoding="utf-8") == (
        "# pyright: reportArgumentType=false, reportCallIssue=false\n"
        "from onnx2tf.tflite_builder.pytorch_package_runtime import load_generated_model_package\n"
    )


def test_common_package_files_preserve_explicit_runtime_source(tmp_path) -> None:
    runtime_source = "VALUE = 17\n"

    _write_generated_package_common_files(
        str(tmp_path),
        runtime_source=runtime_source,
    )

    assert (tmp_path / "runtime.py").read_text(encoding="utf-8") == runtime_source


def test_wrapper_model_source_preserves_public_loader_contract(tmp_path) -> None:
    _write_wrapper_model_file(str(tmp_path))
    model_source = (tmp_path / "model.py").read_text(encoding="utf-8")

    ast.parse(model_source)
    assert "class Model(torch.nn.Module):" in model_source
    assert "def forward_named(self, *args: Any, **kwargs: Any) -> Any:" in model_source
    assert (
        "def load_model(device: str | None = None, eval_mode: bool = True) -> Model:"
        in model_source
    )
    assert "load_generated_model_package" in model_source


def test_pool2d_runtime_recovery_patch_is_ordered_and_idempotent(tmp_path) -> None:
    runtime_path = tmp_path / "runtime.py"
    runtime_path.write_text(
        "def _apply_pool2d(x):\n"
        "    before = True\n"
        "    if resize_as_channel_last and x.ndim == 4 and has_target_shape and len(target) == 4:\n"
        "        return x\n"
        "\n"
        "def later():\n"
        "    pass\n",
        encoding="utf-8",
    )

    _patch_generated_runtime_pool2d_channel_last_recovery(tmp_path)
    first_source = runtime_path.read_text(encoding="utf-8")
    _patch_generated_runtime_pool2d_channel_last_recovery(tmp_path)

    assert runtime_path.read_text(encoding="utf-8") == first_source
    assert first_source.count("int(actual_shape[-1]) == int(target[1])") == 1
    assert first_source.index("if not resize_as_channel_last") < first_source.index(
        "if resize_as_channel_last"
    )


def test_pool2d_runtime_recovery_patch_ignores_missing_or_unmatched_runtime(
    tmp_path,
) -> None:
    _patch_generated_runtime_pool2d_channel_last_recovery(tmp_path)
    runtime_path = tmp_path / "runtime.py"
    runtime_path.write_text("def unrelated():\n    pass\n", encoding="utf-8")

    _patch_generated_runtime_pool2d_channel_last_recovery(tmp_path)

    assert runtime_path.read_text(encoding="utf-8") == "def unrelated():\n    pass\n"


def test_native_runtime_source_normalizes_sequence_annotations() -> None:
    helper_source = (
        "def helper(a: Optional[Sequence[int]], b: Sequence[str], "
        "c: Sequence[torch.Tensor], d: Sequence[Any]) -> Sequence[int]:\n"
        "    return a\n\n"
    )

    runtime_source = _build_native_runtime_source(helper_source)

    ast.parse(runtime_source)
    assert "Optional[Sequence[int]]" not in runtime_source
    assert "Sequence[int]" not in runtime_source
    assert "Sequence[str]" not in runtime_source
    assert "Sequence[torch.Tensor]" not in runtime_source
    assert "Sequence[Any]" not in runtime_source
    assert "Optional[List[int]]" in runtime_source
    assert "List[torch.Tensor]" in runtime_source
    assert "torch.load(package_dir / 'state_dict.pth'" in runtime_source
