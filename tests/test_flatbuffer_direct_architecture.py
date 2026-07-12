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


def test_constant_input_fold_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "constant_fold.py"
    )
    function_names = {
        "_optimize_constant_input_cast_chains",
        "_optimize_constant_input_pad_chains",
        "_optimize_constant_input_pool_chains",
    }

    def _functions(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    assert function_names <= _functions(pass_path)
    assert function_names.isdisjoint(_functions(lowering_path))


def test_boundary_input_layout_pass_and_graph_helpers_have_single_owners() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "boundary_input_layout.py"
    )
    common_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "core"
        / "model_ir_utils.py"
    )
    reporting_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "reporting.py"
    precision_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "precision.py"
    )
    channel_slice_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "channel_slice_layout.py"
    )
    boundary_chains_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "boundary_input_chains.py"
    )
    input_passthrough_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "input_passthrough_layout.py"
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
    assert "_optimize_boundary_input_layout_transposes" in pass_functions
    wrapper = lowering_functions["_optimize_boundary_input_layout_transposes"]
    wrapper_names = {
        node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)
    }
    assert "_optimize_boundary_input_layout_transposes_pass" in wrapper_names
    lowerer_names = {
        node.id
        for node in ast.walk(
            ast.parse(lowering_path.read_text(encoding="utf-8"))
        )
        if isinstance(node, ast.Name)
    }
    assert "run_boundary_input_layout_cleanup" in lowerer_names

    graph_helpers = {
        "_broadcast_static_shapes",
        "_build_tensor_consumer_map",
        "_invert_perm",
        "_is_singleton_constant_tensor",
        "_read_singleton_constant_float",
        "_normalize_squeeze_axes_for_rank",
        "_permute_tensor_metadata_if_rank_matches",
        "_read_const_ints_from_tensor",
        "_read_transpose_perm",
        "_rename_tensor_globally",
        "_replace_operator_input_at",
        "_replace_tensor_inputs",
        "_set_operator_inputs",
        "_set_operator_outputs",
        "_write_const_ints_to_tensor",
    }
    assert graph_helpers <= set(common_functions)
    for path in (lowering_path, reporting_path, precision_path):
        assert graph_helpers.isdisjoint(set(_functions(path)))

    channel_slice_functions = {
        "_optimize_boundary_input_transpose_channel_slice_blocks",
        "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains",
        "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains",
        "_optimize_transpose_channel_slice_dual_add_bridges_strict",
        "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks",
    }
    pass_functions = _functions(channel_slice_path)
    assert channel_slice_functions <= set(pass_functions)
    for function_name in channel_slice_functions:
        wrapper = lowering_functions[function_name]
        wrapper_names = {
            node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names

    boundary_chain_functions = {
        "_optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains",
        "_optimize_boundary_input_transpose_batchmatmul_chains",
    }
    pass_functions = _functions(boundary_chains_path)
    assert boundary_chain_functions <= set(pass_functions)
    for function_name in boundary_chain_functions:
        wrapper = lowering_functions[function_name]
        wrapper_names = {
            node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names

    input_passthrough_functions = {
        "_optimize_asin_transpose_passthrough_chains",
        "_optimize_erf_transpose_passthrough_chains",
        "_optimize_hardsigmoid_transpose_passthrough_chains",
        "_optimize_hardsigmoid_mul_transpose_passthrough_chains",
        "_optimize_hardswish_transpose_passthrough_chains",
        "_optimize_leading_input_transpose_passthrough_chains",
    }
    pass_functions = _functions(input_passthrough_path)
    assert input_passthrough_functions <= set(pass_functions)
    for function_name in input_passthrough_functions:
        wrapper = lowering_functions[function_name]
        wrapper_names = {
            node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names


def test_graph_cleanup_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "graph_cleanup.py"
    )
    function_names = {
        "_optimize_fold_consecutive_mul_constants_chains",
        "_optimize_squeeze_reshape_identity_chains",
        "_optimize_squeeze_unary_reshape_passthrough_chains",
        "_optimize_maximum_minimum_relu0to1_chains",
        "_optimize_maximum_with_zero_input2_to_relu",
        "_optimize_duplicate_reshape_fanout",
        "_optimize_duplicate_transpose_fanout",
    }

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    lowering_functions = _functions(lowering_path)
    assert function_names <= set(_functions(pass_path))
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names
    lowerer_names = {
        node.id
        for node in ast.walk(
            ast.parse(lowering_path.read_text(encoding="utf-8"))
        )
        if isinstance(node, ast.Name)
    }
    assert "run_clamp_cleanup" in lowerer_names
    assert "run_squeeze_reshape_identity_cleanup" in lowerer_names


def test_ordered_model_ir_runner_calls_record_session_diagnostics() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    runner_names = {
        "run_boundary_input_layout_cleanup",
        "run_clamp_cleanup",
        "run_constant_input_fold_cleanup",
        "run_consecutive_mul_constants_cleanup",
        "run_conv_attention_layout_cleanup",
        "run_duplicate_fanout_cleanup",
        "run_mixed_attention_layout_cleanup",
        "run_maximum_zero_relu_cleanup",
        "run_qkv_attention_bridge_cleanup",
        "run_redundant_cast_cleanup",
        "run_squeeze_reshape_identity_cleanup",
        "run_terminal_quantize_dequantize_cleanup",
    }
    tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in runner_names
    ]

    assert {call.func.id for call in calls if isinstance(call.func, ast.Name)} == runner_names
    assert len(calls) == 37
    for call in calls:
        diagnostics_keywords = [
            keyword for keyword in call.keywords if keyword.arg == "diagnostics"
        ]
        assert len(diagnostics_keywords) == 1
        value = diagnostics_keywords[0].value
        assert isinstance(value, ast.Attribute)
        assert value.attr == "diagnostics"
        assert isinstance(value.value, ast.Name)
        assert value.value.id == "session"


def test_cast_cleanup_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "cast_cleanup.py"
    )
    function_names = {
        "_optimize_redundant_int32_to_int64_passthrough_cast_chains",
        "_optimize_redundant_int64_to_int32_cast_chains",
    }

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    lowering_functions = _functions(lowering_path)
    assert function_names <= set(_functions(pass_path))
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names


def test_quantization_cleanup_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantization_cleanup.py"
    )
    function_names = {
        "_optimize_terminal_quantize_dequantize",
        "_quantized_tensors_share_exact_grid",
    }

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    lowering_functions = _functions(lowering_path)
    assert function_names <= set(_functions(pass_path))
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names


def test_attention_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "attention_layout.py"
    )
    function_names = {
        "_optimize_transpose_csp_attention_nhwc_chains",
        "_optimize_transpose_conv_attention_nhwc_propagation_chains",
        "_optimize_attention_qkv_gather_reshape_transpose_hoist_chains",
        "_optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains",
        "_optimize_attention_qkv_shared_pretranspose_slice_nchw_chains",
        "_optimize_attention_qkv_slice_replace_gather_reshape_chains",
        "_optimize_attention_qkv_slice_to_split_chains",
        "_optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains",
    }

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    lowering_functions = _functions(lowering_path)
    assert function_names <= set(_functions(pass_path))
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names
    lowerer_names = {
        node.id
        for node in ast.walk(
            ast.parse(lowering_path.read_text(encoding="utf-8"))
        )
        if isinstance(node, ast.Name)
    }
    assert "run_conv_attention_layout_cleanup" in lowerer_names
    assert "run_mixed_attention_layout_cleanup" in lowerer_names
    assert "run_qkv_attention_bridge_cleanup" in lowerer_names


def test_pad_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "pad_layout.py"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    pass_names = {
        "_optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains",
        "_optimize_transpose_instancenorm_pad_prepost_nhwc_chains",
        "_optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains",
        "_optimize_transpose_pad_mul_posttranspose_add_nhwc_chains",
        "_optimize_transpose_pad_prepost_nhwc_chains",
        "_optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains",
    }
    lowering_functions = _functions(lowering_path)
    assert pass_names <= set(_functions(pass_path))
    for function_name in pass_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names


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
