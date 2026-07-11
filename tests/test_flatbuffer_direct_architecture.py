from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDED_ROOTS = [
    REPO_ROOT / "onnx2tf" / "tflite_builder" / name
    for name in ["core", "passes", "op_families"]
]


def _imports_tensorflow(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "tensorflow" or alias.name.startswith("tensorflow.") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module == "tensorflow" or module.startswith("tensorflow."):
                return True
    return False


def test_flatbuffer_direct_core_modules_are_context_bounded() -> None:
    oversized = {}
    for root in BOUNDED_ROOTS:
        for path in root.glob("*.py"):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > 2000:
                oversized[str(path.relative_to(REPO_ROOT))] = line_count
    assert oversized == {}


def test_flatbuffer_direct_core_has_no_tensorflow_imports() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for root in BOUNDED_ROOTS
        for path in root.glob("*.py")
        if _imports_tensorflow(path)
    ]
    assert offenders == []


def test_legacy_megafiles_cannot_grow_while_they_are_being_retired() -> None:
    # Generated schema is intentionally excluded. These ceilings capture the
    # migration baseline and force new code into bounded modules.
    ceilings = {
        "onnx2tf/tflite_builder/lower_from_onnx2tf.py": 75505,
        "onnx2tf/tflite_builder/op_registry.py": 9030,
        "onnx2tf/tflite_builder/pytorch_exporter.py": 46125,
        "tests/test_tflite_builder_direct.py": 40350,
        "tests/test_pytorch_exporter.py": 47000,
    }
    violations = {}
    for relative, ceiling in ceilings.items():
        count = len((REPO_ROOT / relative).read_text(encoding="utf-8").splitlines())
        if count > ceiling:
            violations[relative] = {"lines": count, "ceiling": ceiling}
    assert violations == {}
