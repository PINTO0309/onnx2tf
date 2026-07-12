from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_SCOPED_ROOTS = [
    REPO_ROOT / "onnx2tf" / "tflite_builder" / name
    for name in ["core", "passes", "op_families"]
]
DEPENDENCY_SCOPED_FILES = [
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "quantized_common.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "dynamic_quantize.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "qlinear_fc.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "qlinear_binary.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "qlinear_activation.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "qlinear_concat.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "quantize_linear.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "qlinear_conv.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "conv_integer.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "op_builders"
    / "qlinear_pool.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_codegen_utils.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_layout_utils.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_utils.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_layout_passes.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_bridge_passes.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_model_passes.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_optimizer.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_export_errors.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_export_support.py",
]

PYTORCH_PURE_UTILITY_FILES = DEPENDENCY_SCOPED_FILES[:-2]


def _dependency_scoped_python_files():
    yield from DEPENDENCY_SCOPED_FILES
    for root in DEPENDENCY_SCOPED_ROOTS:
        yield from root.glob("*.py")


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


def test_flatbuffer_direct_core_has_no_tensorflow_imports() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _dependency_scoped_python_files()
        if _imports_tensorflow(path)
    ]
    assert offenders == []


def test_pytorch_pure_utilities_do_not_import_torch() -> None:
    offenders = []
    for path in PYTORCH_PURE_UTILITY_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [str(node.module or "")]
            if any(module == "torch" or module.startswith("torch.") for module in modules):
                offenders.append(str(path.relative_to(REPO_ROOT)))
                break
    assert offenders == []


def test_dynamic_quantize_builder_stays_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "dynamic_quantize.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "build_dynamic_quantize_linear_op" in family_functions
    assert "build_dynamic_quantize_linear_op" not in legacy_functions


def test_qlinear_fc_builders_stay_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "qlinear_fc.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert {"build_qlinear_matmul_op", "build_qgemm_op"} <= family_functions
    assert "build_qlinear_matmul_op" not in legacy_functions
    assert "build_qgemm_op" not in legacy_functions


def test_qlinear_binary_builders_stay_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "qlinear_binary.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    expected = {"build_qlinear_add_op", "build_qlinear_mul_op"}
    assert expected <= family_functions
    assert expected.isdisjoint(legacy_functions)


def test_qlinear_activation_builders_stay_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "qlinear_activation.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    expected = {
        "build_qlinear_sigmoid_op",
        "build_qlinear_leaky_relu_op",
        "build_qlinear_softmax_op",
    }
    assert expected <= family_functions
    assert expected.isdisjoint(legacy_functions)


def test_qlinear_concat_builder_stays_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "qlinear_concat.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "build_qlinear_concat_op" in family_functions
    assert "build_qlinear_concat_op" not in legacy_functions


def test_quantize_linear_builders_stay_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantize_linear.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    expected = {"build_quantize_linear_op", "build_dequantize_linear_op"}
    assert expected <= family_functions
    assert expected.isdisjoint(legacy_functions)


def test_qlinear_conv_builder_stays_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "qlinear_conv.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "build_qlinear_conv_op" in family_functions
    assert "build_qlinear_conv_op" not in legacy_functions


def test_conv_integer_builder_stays_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "conv_integer.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "build_conv_integer_op" in family_functions
    assert "build_conv_integer_op" not in legacy_functions


def test_qlinear_pool_builders_stay_in_family_module() -> None:
    family_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "qlinear_pool.py"
    )
    legacy_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "op_builders"
        / "quantized_common.py"
    )
    family_source = family_path.read_text(encoding="utf-8")
    legacy_source = legacy_path.read_text(encoding="utf-8")
    family_functions = {
        node.name
        for node in ast.parse(family_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_functions = {
        node.name
        for node in ast.parse(legacy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    expected = {
        "build_qlinear_average_pool_op",
        "build_qlinear_global_average_pool_op",
    }
    assert expected <= family_functions
    assert expected.isdisjoint(legacy_functions)
