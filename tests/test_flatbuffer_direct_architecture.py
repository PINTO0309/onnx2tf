from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_SCOPED_ROOTS = [
    REPO_ROOT / "onnx2tf" / "tflite_builder" / name
    for name in ["core", "passes", "op_families"]
]
DEPENDENCY_SCOPED_FILES = [
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "reporting.py",
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


def test_reporting_implementation_stays_out_of_lowering_module() -> None:
    reporting_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "reporting.py"
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    reporting_tree = ast.parse(reporting_path.read_text(encoding="utf-8"))
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    reporting_functions = {
        node.name
        for node in reporting_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    implementation_functions = {
        "_collect_schema_ops_for_range",
        "_build_schema_policy_matrix",
        "_trace_tensor_rewrite_history",
        "_build_onnx_tensor_consumer_graph",
        "_infer_correspondence_via_downstream",
    }
    assert implementation_functions <= reporting_functions
    assert implementation_functions.isdisjoint(lowering_functions)

    public_delegates = {
        "build_op_coverage_report": "_build_op_coverage_report",
        "write_op_coverage_report": "_write_op_coverage_report",
        "build_tensor_correspondence_report": "_build_tensor_correspondence_report",
        "write_tensor_correspondence_report": "_write_tensor_correspondence_report",
    }
    assert set(public_delegates) <= reporting_functions
    for public_name, delegate_name in public_delegates.items():
        wrapper = lowering_functions[public_name]
        referenced_names = {
            node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)
        }
        assert delegate_name in referenced_names


def test_high_rank_matmul_pass_and_prune_utility_have_single_owners() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "high_rank_matmul.py"
    )
    common_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "core"
        / "model_ir_utils.py"
    )
    precision_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "precision.py"
    )
    constant_fold_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "constant_fold.py"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    lowering_functions = _functions(lowering_path)
    pass_functions = _functions(pass_path)
    common_functions = _functions(common_path)
    assert "_compress_static_high_rank_batch_matmul" in pass_functions
    wrapper = lowering_functions["_compress_static_high_rank_batch_matmul"]
    wrapper_names = {
        node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)
    }
    assert "_compress_static_high_rank_batch_matmul_pass" in wrapper_names

    assert "_prune_unused_tensors" in common_functions
    assert "_is_fully_known_positive_shape" in common_functions
    for path in (lowering_path, precision_path, constant_fold_path):
        functions = _functions(path)
        assert "_prune_unused_tensors" not in functions
        assert "_is_fully_known_positive_shape" not in functions


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
