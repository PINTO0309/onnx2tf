from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder._pytorch_exporter_native_codegen_pipeline import (
    _NATIVE_CODEGEN_FUNCTION_SOURCE,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_SCOPED_ROOTS = [
    REPO_ROOT / "onnx2tf" / "tflite_builder" / name
    for name in ["core", "passes", "op_families"]
]
DEPENDENCY_SCOPED_FILES = [
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "artifact_preparation.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "artifact_metadata.py",
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
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_codegen_values.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_capabilities.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_layout_utils.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_indexing_codegen.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_naming.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_package_sources.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_package_selection.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "pytorch_runtime_wrapper_exporter.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_utils.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_layout_passes.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_bridge_passes.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_model_passes.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_onnx_optimizer.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "pytorch_onnx_artifact_support.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_codegen_stages.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_shape_policy.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "pytorch_state_dict_support.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "pytorch_string_normalizer_exporter.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "pytorch_source_graph_rewrites.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_source_parser.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_source_rewrites.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "pytorch_exported_program_archive.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "pytorch_exported_program_child.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_emitters.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_export_errors.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_export_support.py",
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_artifact_exporters.py",
]

PYTORCH_PURE_UTILITY_FILES = [
    path
    for path in DEPENDENCY_SCOPED_FILES
    if path.name
    not in {
        "pytorch_artifact_exporters.py",
        "pytorch_export_support.py",
        "pytorch_exported_program_archive.py",
        "pytorch_runtime_wrapper_exporter.py",
        "pytorch_state_dict_support.py",
    }
]


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


def test_op_builders_mutate_layout_only_through_lowering_context() -> None:
    op_builders_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "op_builders"
    offenders: list[str] = []
    for path in op_builders_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and node.attr in {"logical_layout", "physical_layout"}
            ):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.attr}"
                )

    assert offenders == []


def test_op_builders_mutate_operator_list_only_through_lowering_context() -> None:
    op_builders_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "op_builders"
    mutating_methods = {"append", "extend", "insert", "pop", "remove", "clear"}
    offenders: list[str] = []

    def _is_context_operator_list(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "operators"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "model_ir"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ctx"
        )

    for path in op_builders_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_mutating_call = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in mutating_methods
                and _is_context_operator_list(node.func.value)
            )
            is_direct_store = (
                isinstance(node, (ast.Attribute, ast.Subscript))
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and _is_context_operator_list(
                    node if isinstance(node, ast.Attribute) else node.value
                )
            )
            if is_mutating_call or is_direct_store:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                )

    assert offenders == []


def test_lowerer_layout_recovery_prefix_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_layout_recovery_prefix_pass_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_transpose_quant_dequant_bridges",
        "_run_boundary_batchmatmul_unary_layout_pass_cluster",
        "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains",
        "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
        "run_hard_activation_passthrough_cleanup",
        "_optimize_swish_transpose_passthrough_chains",
        "_optimize_gelu_tanh_transpose_passthrough_chains",
        "_optimize_center_size_offset_terminal_transpose_chains",
        "_optimize_leakyrelu_transpose_passthrough_chains",
        "_optimize_prelu_transpose_passthrough_chains",
        "_optimize_transpose_elementwise_concat_conv_nhwc_groups",
        "run_spp_layout_cleanup",
        "_optimize_transpose_pre_concat_nhwc_chains",
        "run_ndhwc_concat_layout_cleanup",
        "_optimize_transpose_stridedslice_pre_concat_nhwc_chains",
        "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
        "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
        "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains",
        "_run_channel_shuffle_gather_layout_pass_cluster",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    registered_calls = {
        call.func.id: call
        for call in helper_calls
        if call.func.id
        in {
            "run_hard_activation_passthrough_cleanup",
            "run_spp_layout_cleanup",
            "run_ndhwc_concat_layout_cleanup",
        }
    }
    assert set(registered_calls) == {
        "run_hard_activation_passthrough_cleanup",
        "run_spp_layout_cleanup",
        "run_ndhwc_concat_layout_cleanup",
    }
    for call in registered_calls.values():
        assert all(keyword.arg != "state_scope" for keyword in call.keywords)

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 2


def test_transpose_qdq_bridge_optimizer_has_one_module_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper_name = "_optimize_transpose_quant_dequant_bridges"
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_optimize_transpose_quant_dequant_bridges_pass"
    ]
    assert len(dispatches) == 1

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    recovery_prefix = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_layout_recovery_prefix_pass_sequence"
    )
    production_calls = [
        node
        for node in ast.walk(recovery_prefix)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "transpose_qdq_bridge_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "optimize_transpose_quant_dequant_bridges"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    for pattern_name in ("Pattern A:", "Pattern B:", "Pattern C:", "Pattern D:"):
        assert pattern_name in owner_function_source
    assert "while True:" in owner_function_source
    assert "_build_tensor_consumer_map(model_ir)" in owner_function_source
    assert "_build_tensor_producer_map(model_ir)" in owner_function_source
    assert "_prune_unused_tensors(model_ir)" in owner_function_source
    for stat_key in (
        "removed_transpose_quantize_dequantize_bridges",
        "rewritten_add_qdq_residual_transpose_bridges",
        "rewritten_mixed_add_qdq_residual_transpose_bridges",
    ):
        assert f'"{stat_key}"' in owner_function_source
    assert "lower_from_onnx2tf" not in owner_source


def test_lowerer_layout_reshape_attention_prefix_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_layout_reshape_attention_recovery_prefix"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_run_layout_recovery_prefix_pass_sequence",
        "_optimize_transpose_pre_add_nhwc_chains",
        "_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains",
        "_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains",
        "_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains",
        "_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains",
        "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains",
        "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains",
        "_optimize_attention_gather_transpose_reshape_cleanup_chains",
        "_optimize_gather_axis0_singleton_to_reshape_input_chains",
        "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains",
        "_optimize_window_partition_reshape_transpose_to_space_to_depth_chains",
        "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains",
        "_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains",
        "run_squeeze_reshape_identity_cleanup",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    squeeze_call = helper_calls[-1]
    assert all(keyword.arg != "state_scope" for keyword in squeeze_call.keywords)
    include_unary = next(
        keyword
        for keyword in squeeze_call.keywords
        if keyword.arg == "include_unary_passthrough"
    )
    assert isinstance(include_unary.value, ast.Constant)
    assert include_unary.value.value is True

    layout_recovery = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
            for node in ast.walk(statement)
        )
        == 3
    )
    invocation_indexes = [
        index
        for index, statement in enumerate(layout_recovery.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    ]
    assert len(invocation_indexes) == 3
    next_call_names = []
    for index in invocation_indexes:
        invocation = layout_recovery.body[index].value
        assert invocation.args == []
        assert invocation.keywords == []
        following = layout_recovery.body[index + 1]
        assert isinstance(following, ast.Expr)
        assert isinstance(following.value, ast.Call)
        assert isinstance(following.value.func, ast.Name)
        next_call_names.append(following.value.func.id)
    assert next_call_names == [
        "_optimize_fold_mul_add_mul_affine_chains",
        "_optimize_fold_mul_add_mul_affine_chains",
        "_optimize_transpose_instancenorm_prepost_nhwc_chains",
    ]


def test_lowerer_preadd_mean_attention_recovery_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_preadd_mean_attention_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_transpose_pre_add_nhwc_chains",
        "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
        "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
        "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
        "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains",
        "_run_mean_attention_layout_pass_cluster",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    recovery_block = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
            for node in ast.walk(statement)
        )
        == 2
    )
    invocation_indexes = [
        index
        for index, statement in enumerate(recovery_block.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    ]
    assert len(invocation_indexes) == 2
    previous_call_names = []
    next_call_names = []
    for index in invocation_indexes:
        invocation = recovery_block.body[index].value
        assert invocation.args == []
        assert invocation.keywords == []
        previous = recovery_block.body[index - 1]
        following = recovery_block.body[index + 1]
        for boundary in (previous, following):
            assert isinstance(boundary, ast.Expr)
            assert isinstance(boundary.value, ast.Call)
            assert isinstance(boundary.value.func, ast.Name)
        previous_call_names.append(previous.value.func.id)
        next_call_names.append(following.value.func.id)
    assert previous_call_names == [
        "_run_layout_recovery_prefix_pass_sequence",
        "_run_channel_shuffle_gather_layout_pass_cluster",
    ]
    assert next_call_names == [
        "_run_attention_gate_qdq_recovery_sequence",
        "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",
    ]


def test_lowerer_attention_gate_qdq_recovery_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_attention_gate_qdq_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",
        "_optimize_sinet_mix_attention_double_logistic_nhwc_chains",
        "_run_gate_layout_pass_cluster",
        "_optimize_transposeconv_output_nhwc_passthrough_chains",
        "_optimize_transposeconv_output_channel1_terminal_transpose_chains",
        "_run_transpose_unary_fanout_layout_pass_cluster",
        "_optimize_transpose_dequant_relu_quantize_bridges",
        "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
        "run_trailing_output_transpose_cleanup",
        "_optimize_transpose_dequant_mul_add_prelu_quantize_bridges",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order
    trailing_call = helper_calls[-2]
    assert all(keyword.arg != "state_scope" for keyword in trailing_call.keywords)

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 3

    outer_helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_layout_attention_quantized_recovery_suffix"
    )
    outer_index = next(
        index
        for index, statement in enumerate(outer_helper.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    assert outer_helper.body[outer_index - 1].value.func.id == (
        "_run_mean_attention_layout_pass_cluster"
    )
    assert outer_helper.body[outer_index + 1].value.func.id == (
        "_run_duplicate_quantized_prelu_pass_cluster"
    )

    direct_boundaries = []
    for statement in lowerer.body:
        if not (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Name)
            and statement.test.id == "optimize_layout_transpose_chains"
        ):
            continue
        for index, candidate in enumerate(statement.body):
            if not (
                isinstance(candidate, ast.Expr)
                and isinstance(candidate.value, ast.Call)
                and isinstance(candidate.value.func, ast.Name)
                and candidate.value.func.id == helper_name
            ):
                continue
            direct_boundaries.append(
                (statement.body[index - 1].value, statement.body[index + 1].value)
            )
    assert len(direct_boundaries) == 2
    assert [previous.func.id for previous, _ in direct_boundaries] == [
        "_run_mean_attention_layout_pass_cluster",
        "_run_preadd_mean_attention_recovery_sequence",
    ]
    assert [following.func.id for _, following in direct_boundaries] == [
        "run_quantized_prelu_cleanup",
        "_optimize_dequant_transposeconv_quantize_chains",
    ]
    assert any(
        keyword.arg == "include_layernorm"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in direct_boundaries[0][0].keywords
    )
    assert direct_boundaries[1][0].keywords == []


def test_lowerer_quantized_activation_binary_recovery_has_one_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_quantized_activation_binary_bridge_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_dequant_hardsigmoid_quantize_chains",
        "_optimize_dequant_maxpool_quantize_chains",
        "_optimize_dequant_softmax_quantize_chains",
        "_optimize_dequant_logistic_quantize_chains",
        "_canonicalize_softmax_transpose_chains",
        "_run_safe_binary_bridge_recovery_sequence",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    direct_boundaries = []
    for statement in lowerer.body:
        if not (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Name)
            and statement.test.id == "optimize_layout_transpose_chains"
        ):
            continue
        for index, candidate in enumerate(statement.body):
            if not (
                isinstance(candidate, ast.Expr)
                and isinstance(candidate.value, ast.Call)
                and isinstance(candidate.value.func, ast.Name)
                and candidate.value.func.id == helper_name
            ):
                continue
            assert candidate.value.args == []
            assert candidate.value.keywords == []
            direct_boundaries.append(
                (statement.body[index - 1], statement.body[index + 1])
            )
    assert len(direct_boundaries) == 2
    previous_call_names = []
    for previous, _ in direct_boundaries:
        assert isinstance(previous, ast.Expr)
        assert isinstance(previous.value, ast.Call)
        assert isinstance(previous.value.func, ast.Name)
        previous_call_names.append(previous.value.func.id)
    assert previous_call_names == [
        "run_quantized_reshape_cleanup",
        "_optimize_dequant_transposeconv_quantize_chains",
    ]

    first_following = direct_boundaries[0][1]
    assert isinstance(first_following, ast.If)
    assert isinstance(first_following.test, ast.Name)
    assert (
        first_following.test.id
        == "enable_transpose_binary_bridge_optimizations"
    )
    second_following = direct_boundaries[1][1]
    assert isinstance(second_following, ast.Expr)
    assert isinstance(second_following.value, ast.Call)
    assert isinstance(second_following.value.func, ast.Name)
    assert (
        second_following.value.func.id
        == "_optimize_transpose_elementwise_concat_conv_nhwc_groups"
    )


def test_lowerer_safe_binary_bridge_recovery_has_one_ordered_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "binary_bridge_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_safe_binary_bridge_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_phases = [
        "_SAFE_LEGACY_PHASE",
        "_SAFE_SINGLE_POST_PHASE",
        "_SAFE_MIXED_FANOUT_PHASE",
        "_SAFE_ASYMMETRIC_FANOUT_PHASE",
        "_SAFE_FULL_POST_FANOUT_PHASE",
    ]
    phase_assignment = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_SAFE_PHASES"
            for target in node.targets
        )
    )
    assert isinstance(phase_assignment.value, ast.Tuple)
    assert [
        element.id
        for element in phase_assignment.value.elts
        if isinstance(element, ast.Name)
    ] == expected_phases
    assert "def _resolve_multi_post(" in owner_source
    assert "def _resolve_asymmetric_fanout(" in owner_source
    assert "def _apply_multi_post(" in owner_source
    assert "def _apply_asymmetric_fanout(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source

    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == [
        "_run_safe_binary_bridge_recovery_pass"
    ]
    layout_keyword = next(
        keyword
        for keyword in helper_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"

    compatibility_dispatchers = {
        "_optimize_transpose_binary_symmetric_legacy_only_bridges_safe":
            "_optimize_transpose_binary_symmetric_legacy_only_bridges_safe_pass",
        "_optimize_transpose_binary_single_post_bridges_safe":
            "_optimize_transpose_binary_single_post_bridges_safe_pass",
        "_optimize_transpose_binary_mixed_fanout_bridges_safe":
            "_optimize_transpose_binary_mixed_fanout_bridges_safe_pass",
        "_optimize_transpose_binary_asymmetric_fanout_bridges":
            "_optimize_transpose_binary_asymmetric_fanout_bridges_pass",
        "_optimize_transpose_binary_full_post_fanout_bridges":
            "_optimize_transpose_binary_full_post_fanout_bridges_pass",
    }
    for wrapper_name, pass_name in compatibility_dispatchers.items():
        wrapper = next(
            node
            for node in lowering_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        assert len(wrapper.body) == 2
        dispatch = wrapper.body[1]
        assert isinstance(dispatch, ast.Return)
        call = next(
            node for node in ast.walk(dispatch) if isinstance(node, ast.Call)
        )
        assert isinstance(call.func, ast.Name)
        assert call.func.id == pass_name
        assert {keyword.arg for keyword in call.keywords} == {
            "graph_index",
            "layout_state",
            "max_rewrites",
            "candidate",
        }

    quantized_helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_quantized_activation_binary_bridge_recovery_sequence"
    )
    assert quantized_helper.body[-1].value.func.id == helper_name

    direct_boundaries = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if not (
                isinstance(candidate, ast.Expr)
                and isinstance(candidate.value, ast.Call)
                and isinstance(candidate.value.func, ast.Name)
                and candidate.value.func.id == helper_name
            ):
                continue
            direct_boundaries.append(
                (statement.body[index - 1], statement.body[index + 1])
            )
    assert len(direct_boundaries) == 2
    assert [boundary[0].value.func.id for boundary in direct_boundaries] == [
        "_run_layout_attention_quantized_recovery_suffix",
        "_run_transpose_unary_fanout_layout_pass_cluster",
    ]
    assert [boundary[1].value.func.id for boundary in direct_boundaries] == [
        "_optimize_transpose_dequantize_mean_quantize_bridges",
        "_advance_post_progress",
    ]
    all_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(all_invocations) == 3


def test_lowerer_qlinear_mean_concat_recovery_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_qlinear_mean_concat_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_transpose_mean_hardsigmoid_muladd_chains",
        "_optimize_nhwc_prefix_qlinear_silu_chains",
        "_optimize_nhwc_propagation_qlinear_concat_conv",
        "_optimize_concat_pre_quantize_dequantize",
        "_optimize_transpose_mean_maxpool_concat_conv_chains",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    boundaries = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if not (
                isinstance(candidate, ast.Expr)
                and isinstance(candidate.value, ast.Call)
                and isinstance(candidate.value.func, ast.Name)
                and candidate.value.func.id == helper_name
            ):
                continue
            boundaries.append(
                (statement.body[index - 1], statement.body[index + 1])
            )
    assert len(boundaries) == 2
    assert [boundary[0].value.func.id for boundary in boundaries] == [
        "_optimize_transpose_dequantize_mean_quantize_bridges",
        "_set_post_progress_desc",
    ]
    assert [boundary[1].value.func.id for boundary in boundaries] == [
        "_run_layout_reshape_attention_recovery_prefix",
        "_run_layout_recovery_prefix_pass_sequence",
    ]


def test_qlinear_silu_prefix_corrected_owner_contract_is_explicit() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "qlinear_silu_prefix_layout.py"
    )
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    owner_name = "_optimize_nhwc_prefix_qlinear_silu_chains"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert owner.end_lineno - owner.lineno + 1 == 513
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" in call_names
    assert "_build_tensor_producer_map" not in call_names
    assert "ModelIRGraphIndex" not in call_names
    assert "_read_transpose_perm" in call_names
    assert "_set_operator_inputs" in call_names
    assert "_replace_tensor_inputs" in call_names
    assert "_permute_tensor_metadata_if_rank_matches" in call_names
    assert "_prune_unused_tensors" in call_names
    assert "_is_exact_nhwc_to_nchw_perm_tensor" in call_names
    assert "_unique_tensor_name" in call_names
    outer_loops = [node for node in owner.body if isinstance(node, ast.While)]
    assert len(outer_loops) == 1
    assert isinstance(outer_loops[0].test, ast.Constant)
    assert outer_loops[0].test.value is True

    metadata_plan = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "metadata_target_names"
            for target in node.targets
        )
    )
    signature_plan = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "legacy_mul_signature"
            for target in node.targets
        )
    )
    adapter_plan = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "adapter_insertions"
    )
    tensor_mutations = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and "model_ir.tensors" in ast.unparse(target)
            for target in node.targets
        )
    ]
    setter_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_operator_inputs"
    ]
    first_mutation_line = min(
        [node.lineno for node in tensor_mutations]
        + [node.lineno for node in setter_calls]
    )
    assert metadata_plan.lineno < first_mutation_line
    assert signature_plan.lineno < first_mutation_line
    assert adapter_plan.lineno < first_mutation_line

    prune_guard = next(
        node
        for node in owner.body
        if isinstance(node, ast.If)
        and any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "_prune_unused_tensors"
            for candidate in ast.walk(node)
        )
    )
    assert ast.unparse(prune_guard.test) == "optimized > 0"

    mul_users_assignment = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mul_users"
            for target in node.targets
        )
    )
    assert "dict.fromkeys" in ast.unparse(mul_users_assignment.value)
    legacy_user_loop = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "user_idx"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "mul_users"
    )
    assert mul_users_assignment.lineno < legacy_user_loop.lineno

    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    dispatch = wrapper.body[0].value
    assert isinstance(dispatch, ast.Call)
    assert isinstance(dispatch.func, ast.Name)
    assert dispatch.func.id == f"{owner_name}_pass"
    assert len(dispatch.args) == 1
    assert isinstance(dispatch.args[0], ast.Name)
    assert dispatch.args[0].id == "model_ir"


def test_lowerer_layout_attention_quantized_suffix_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_layout_attention_quantized_recovery_suffix"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
        "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
        "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains",
        "_run_mean_attention_layout_pass_cluster",
        "_run_attention_gate_qdq_recovery_sequence",
        "_run_duplicate_quantized_prelu_pass_cluster",
        "_optimize_dequant_transposeconv_quantize_chains",
        "run_quantized_reshape_cleanup",
        "_optimize_dequant_hardsigmoid_quantize_chains",
        "_optimize_dequant_maxpool_quantize_chains",
        "_optimize_dequant_softmax_quantize_chains",
        "_optimize_dequant_logistic_quantize_chains",
        "_canonicalize_softmax_transpose_chains",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    calls_by_name = {call.func.id: call for call in helper_calls}
    for registered_name in ("run_quantized_reshape_cleanup",):
        assert all(
            keyword.arg != "state_scope"
            for keyword in calls_by_name[registered_name].keywords
        )
    duplicate_keyword = next(
        keyword
        for keyword in calls_by_name[
            "_run_duplicate_quantized_prelu_pass_cluster"
        ].keywords
        if keyword.arg == "include_transpose"
    )
    assert isinstance(duplicate_keyword.value, ast.Name)
    assert duplicate_keyword.value.id == "include_duplicate_transpose"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 2
    for invocation in helper_invocations:
        keyword = next(
            keyword
            for keyword in invocation.keywords
            if keyword.arg == "include_duplicate_transpose"
        )
        assert isinstance(keyword.value, ast.Name)
        assert (
            keyword.value.id
            == "enable_duplicate_transpose_fanout_optimizations"
        )

    layernorm_variant_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_mean_attention_layout_pass_cluster"
        and any(
            keyword.arg == "include_layernorm"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    assert len(layernorm_variant_calls) == 1


def test_lowerer_mean_attention_cluster_reuses_one_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_mean_attention_layout_pass_cluster"
    )
    expected_order = [
        "run_transpose_mean_passthrough_cleanup",
        "run_mean_mul_add_conv_layout_cleanup",
        "run_layernorm_statistics_layout_cleanup",
        "run_terminal_mean_layout_cleanup",
        "run_se_conv_layout_cleanup",
        "run_se_fc_layout_cleanup",
        "run_conv_attention_layout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_mean_attention_layout_pass_cluster"
    ]
    assert len(helper_invocations) == 4


def test_lowerer_qkv_attention_pair_reuses_one_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_qkv_attention_layout_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_layout_transpose_cleanup",
        "run_qkv_attention_prefix_cleanup",
        "run_qkv_attention_bridge_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 3
    late_bridge_invocations = [
        call
        for call in helper_invocations
        if any(
            keyword.arg == "include_prefix"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
    ]
    assert len(late_bridge_invocations) == 1
    layout_keyword = next(
        keyword
        for keyword in late_bridge_invocations[0].keywords
        if keyword.arg == "include_layout_transpose"
    )
    assert isinstance(layout_keyword.value, ast.Name)
    assert layout_keyword.value.id == "optimize_layout_transpose_chains"


def test_lowerer_terminal_slice_concat_recovery_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_terminal_slice_concat_layout_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_run_channel_slice_pad_mul_layout_pass_cluster",
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains",
        "_optimize_concat_mul_add_transpose_nhwc_bridge_chains",
        "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
        "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
        "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains",
        "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks",
        "_optimize_transpose_unary_split_concat_single_post_nchw",
        "_optimize_transpose_split_channelwise_tail_to_single_post_nchw",
        "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw",
        "_sanitize_probable_nhwc_axis_sensitive_ops",
        "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
        "_optimize_transpose_pre_add_nhwc_chains",
        "run_layout_transpose_cleanup",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order
    assert all(
        keyword.arg != "state_scope" for keyword in helper_calls[-1].keywords
    )

    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    ]
    assert len(invocation_indexes) == 2
    previous_keyword_names = []
    next_call_names = []
    for index in invocation_indexes:
        invocation = lowerer.body[index].value
        assert invocation.args == []
        assert invocation.keywords == []
        previous = lowerer.body[index - 1]
        assert isinstance(previous, ast.Expr)
        assert isinstance(previous.value, ast.Call)
        assert isinstance(previous.value.func, ast.Name)
        assert (
            previous.value.func.id
            == "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains"
        )
        previous_keyword_names.append(
            [keyword.arg for keyword in previous.value.keywords]
        )
        following = lowerer.body[index + 1]
        assert isinstance(following, ast.Expr)
        assert isinstance(following.value, ast.Call)
        assert isinstance(following.value.func, ast.Name)
        next_call_names.append(following.value.func.id)
    assert previous_keyword_names == [["layout_state"], []]
    assert next_call_names == [
        "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks",
        "_optimize_transpose_slice_prepost_nhwc_passthrough_chains",
    ]


def test_lowerer_terminal_affine_concat_split_recovery_has_one_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_terminal_affine_concat_split_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_fold_mul_add_mul_affine_chains",
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
        "_optimize_concat_mul_add_transpose_nhwc_bridge_chains",
        "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
        "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
        "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains",
        "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks",
        "_optimize_transpose_unary_split_concat_single_post_nchw",
        "_optimize_transpose_split_channelwise_tail_to_single_post_nchw",
        "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw",
        "_sanitize_probable_nhwc_axis_sensitive_ops",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    ]
    assert len(invocation_indexes) == 2
    previous_call_names = []
    next_call_names = []
    for index in invocation_indexes:
        invocation = lowerer.body[index].value
        assert invocation.args == []
        assert invocation.keywords == []
        previous = lowerer.body[index - 1]
        following = lowerer.body[index + 1]
        for boundary in (previous, following):
            assert isinstance(boundary, ast.Expr)
            assert isinstance(boundary.value, ast.Call)
            assert isinstance(boundary.value.func, ast.Name)
        previous_call_names.append(previous.value.func.id)
        next_call_names.append(following.value.func.id)
    assert previous_call_names == [
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
        "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
    ]
    assert next_call_names == [
        "_optimize_transpose_pre_add_nhwc_chains",
        "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
    ]


def test_lowerer_sinet_preadd_resize_recovery_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_sinet_preadd_resize_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
        "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
        "_optimize_sinet_concat_resize_affine_transpose_chains",
        "_optimize_sinet_dual_resize_affine_transpose_chains",
        "_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains",
        "_optimize_sinet_softmax_mask_residual_nhwc_tail_chains",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    terminal_helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_sinet_terminal_layout_recovery_sequence"
    )
    nested_index = next(
        index
        for index, statement in enumerate(terminal_helper.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    assert terminal_helper.body[nested_index - 1].value.func.id == (
        "_optimize_sinet_shuffle_residual_transpose_chains"
    )
    assert terminal_helper.body[nested_index + 1].value.func.id == (
        "_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains"
    )

    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    ]
    assert len(invocation_indexes) == 3
    previous_call_names = []
    next_call_names = []
    for index in invocation_indexes:
        invocation = lowerer.body[index].value
        assert invocation.args == []
        assert invocation.keywords == []
        previous = lowerer.body[index - 1]
        following = lowerer.body[index + 1]
        for boundary in (previous, following):
            assert isinstance(boundary, ast.Expr)
            assert isinstance(boundary.value, ast.Call)
            assert isinstance(boundary.value.func, ast.Name)
        previous_call_names.append(previous.value.func.id)
        next_call_names.append(following.value.func.id)
    assert previous_call_names == [
        "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
        "_run_sinet_terminal_layout_recovery_sequence",
        "_reconcile_static_tensor_shapes",
    ]
    assert next_call_names == [
        "_run_singleton_reshape_layout_pass_cluster",
        "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
        "_optimize_transpose_csp_attention_nhwc_chains",
    ]

    all_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(all_invocations) == 4


def test_lowerer_sinet_terminal_layout_recovery_has_one_ordered_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_sinet_terminal_layout_recovery_sequence"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "_optimize_sinet_shuffle_residual_transpose_chains",
        "_run_sinet_preadd_resize_recovery_sequence",
        "_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains",
    ]
    helper_calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in helper_calls] == expected_order

    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    ]
    assert len(invocation_indexes) == 2
    previous_call_names = []
    next_call_names = []
    for index in invocation_indexes:
        invocation = lowerer.body[index].value
        assert invocation.args == []
        assert invocation.keywords == []
        previous = lowerer.body[index - 1]
        following = lowerer.body[index + 1]
        for boundary in (previous, following):
            assert isinstance(boundary, ast.Expr)
            assert isinstance(boundary.value, ast.Call)
            assert isinstance(boundary.value.func, ast.Name)
        previous_call_names.append(previous.value.func.id)
        next_call_names.append(following.value.func.id)
    assert previous_call_names == [
        "_run_terminal_clamp_unary_relu_pass_cluster",
        "_run_indexed_shape_convergence_cleanup",
    ]
    assert next_call_names == [
        "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains",
        "_run_sinet_preadd_resize_recovery_sequence",
    ]


def test_terminal_affine_prelu_owner_has_one_lowerer_adapter() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_affine_prelu_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = (
        "optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains"
    )
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert len(
        [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
        ]
    ) == 1

    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_pass"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1


def test_mean_affine_prepost_owner_has_one_lowerer_adapter() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "mean_affine_prepost_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "optimize_transpose_mean_mul_add_const_prepost_nhwc_chains"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert len(
        [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
        ]
    ) == 1

    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_pass"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3


def test_batchmatmul_affine_input_owner_has_one_lowerer_adapter() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "batchmatmul_affine_input_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "optimize_batchmatmul_affine_transpose_input_chains"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert len(
        [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
        ]
    ) == 1

    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_pass"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2


def test_batchmatmul_se_owner_has_one_lowerer_adapter() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "batchmatmul_se_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "optimize_batchmatmul_reshape_se_nhwc_chains"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert len(
        [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
        ]
    ) == 1

    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_pass"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2


def test_batchmatmul_adjoint_owner_has_one_lowerer_adapter() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "batchmatmul_adjoint_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "optimize_batchmatmul_transpose_input_to_adj_flags"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_read_transpose_perm" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert len(
        [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
        ]
    ) == 1

    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_pass"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2


def test_probable_nhwc_axis_sanitizer_has_one_lowerer_adapter() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "probable_nhwc_axis_sanitizer.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "sanitize_probable_nhwc_axis_sensitive_ops"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_write_const_ints_to_tensor" in owner_source
    assert "_set_operator_outputs" in owner_source
    assert any(isinstance(node, ast.While) for node in ast.walk(owner))

    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_pass"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2


def test_elementwise_fanout_owner_has_one_lowerer_adapter() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "elementwise_fanout_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_find_unbound_nonconstant_operator_inputs" in owner_source
    assert "candidate_snapshot = copy.deepcopy(model_ir)" in owner_source
    assert any(isinstance(node, ast.While) for node in ast.walk(owner))

    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_pass"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3


def test_lowerer_indexed_shape_convergence_has_one_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    helper_name = "_run_indexed_shape_convergence_cleanup"
    helper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in calls] == [
        "_prune_dead_operators",
        "_reconcile_static_tensor_shapes",
        "_resolve_dynamic_reshape_shapes",
        "_reconcile_static_tensor_shapes",
    ]
    assert len(
        [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ModelIRGraphIndex"
        ]
    ) == 1
    for call in calls:
        graph_index_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "graph_index"
        )
        assert isinstance(graph_index_keyword.value, ast.Name)
        assert graph_index_keyword.value.id == "graph_index"

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    ]
    assert len(invocation_indexes) == 1
    invocation_index = invocation_indexes[0]
    invocation = lowerer.body[invocation_index].value
    assert len(invocation.args) == 1
    assert isinstance(invocation.args[0], ast.Name)
    assert invocation.args[0].id == "model_ir"
    layout_keyword = next(
        keyword
        for keyword in invocation.keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"
    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    assert previous.value.func.id == "_run_singleton_reshape_layout_pass_cluster"
    assert following.value.func.id == "_run_sinet_terminal_layout_recovery_sequence"


def test_dynamic_reshape_resolution_has_one_module_owner() -> None:
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_dynamic_reshape_shapes"
    )
    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_dynamic_reshape_shapes_pass"
    ]
    assert len(wrapper_calls) == 1
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "graph_index"
    }

    static_wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_reshape_new_shape_from_static_input"
    )
    static_calls = [
        node
        for node in ast.walk(static_wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_reshape_new_shape_from_static_input_pass"
    ]
    assert len(static_calls) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "dynamic_reshape_resolution.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_dynamic_reshape_shapes"
    )
    owner_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "operator_indices" in owner_call_names
    assert "_build_tensor_consumer_map" not in owner_call_names
    assert "_build_tensor_producer_map" not in owner_call_names
    assert "lower_from_onnx2tf" not in owner_source


def test_static_shape_reconciliation_has_one_module_owner() -> None:
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_targets = {
        "_infer_slice_output_shape_and_resolved_params": (
            "_infer_slice_output_shape_and_resolved_params"
        ),
        "_infer_slice_output_signature": "_infer_slice_output_signature",
        "_infer_batch_matmul_output_shape_and_signature": (
            "_infer_batch_matmul_output_shape_and_signature"
        ),
        "_infer_rank4_signature_from_input": (
            "_infer_rank4_signature_from_input"
        ),
        "_normalize_reduce_axes_for_rank": "_normalize_reduce_axes_for_rank",
        "_infer_reduce_output_shape_and_signature": (
            "_infer_reduce_output_shape_and_signature"
        ),
        "_parse_axes_option": "_parse_axes_option",
        "_infer_squeeze_output_shape_and_signature": (
            "_infer_squeeze_output_shape_and_signature"
        ),
        "_infer_conv_out_dim": "_infer_conv_out_dim",
        "_reconcile_static_tensor_shapes": "reconcile_static_tensor_shapes",
    }
    for wrapper_name, target_name in wrapper_targets.items():
        wrapper = next(
            node
            for node in lowerer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        dispatches = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_static_shape_reconciliation_pass"
        ]
        assert len(dispatches) == 1
        assert dispatches[0].func.attr == target_name

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "static_shape_reconciliation.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "reconcile_static_tensor_shapes"
    )
    assert "max_passes = 32" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "lower_from_onnx2tf" not in owner_source
    owner_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_update_tensor_shape" in owner_call_names
    assert "operator_indices" not in owner_call_names


def test_hardswish_shape_sanitizer_has_one_indexed_module_owner() -> None:
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_sanitize_hardswish_tensor_shapes"
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sanitize_hardswish_tensor_shapes_pass"
    ]
    assert len(dispatches) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "hardswish_shape_sanitization.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "sanitize_hardswish_tensor_shapes"
    )
    owner_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "operator_indices" in owner_call_names
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "lower_from_onnx2tf" not in owner_source


def test_squeeze_shape_sanitizer_has_one_module_owner() -> None:
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_sanitize_squeeze_axes_with_static_input_shapes"
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sanitize_squeeze_axes_with_static_input_shapes_pass"
    ]
    assert len(dispatches) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "squeeze_shape_sanitization.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "sanitize_squeeze_axes_with_static_input_shapes"
    )
    owner_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_parse_axes_option" in owner_call_names
    assert "_infer_squeeze_output_shape_and_signature" in owner_call_names
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "lower_from_onnx2tf" not in owner_source


def test_static_shape_signature_sanitizer_has_one_module_owner() -> None:
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)
    wrapper_name = "_sanitize_static_shape_signature_consistency"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_sanitize_static_shape_signature_consistency_pass"
    ]
    assert len(dispatches) == 1

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "static_shape_signature_sanitization.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "sanitize_static_shape_signature_consistency"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    assert "_build_tensor_producer_map(model_ir)" in owner_function_source
    assert "_read_const_ints_from_tensor(" in owner_function_source
    assert "_is_fully_known_positive_shape(" in owner_function_source
    assert "dynamic_lineage_cache" in owner_function_source
    assert "dynamic_lineage_visiting" in owner_function_source
    for operator_type in ("WHERE", "RANGE", "RESHAPE", "TOPK_V2"):
        assert f'"{operator_type}"' in owner_function_source
    for stat_key in (
        "sanitized_static_shape_signature_consistency",
        "preserved_dynamic_boundary_shape_signature",
        "preserved_dynamic_leading_axis_shape_signature",
        "preserved_dynamic_lineage_shape_signature",
    ):
        assert f'"{stat_key}"' in owner_function_source
    assert "lower_from_onnx2tf" not in owner_source


def test_boundary_signature_realigner_has_one_module_owner() -> None:
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)
    wrapper_name = "_realign_dynamic_boundary_shape_signature_map"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_realign_dynamic_boundary_shape_signature_map_pass"
    ]
    assert len(dispatches) == 1

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "static_shape_signature_sanitization.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "realign_dynamic_boundary_shape_signature_map"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    assert (
        "_align_boundary_signature_to_current_shape("
        in owner_function_source
    )
    assert (
        '"dynamic_boundary_shape_signature_map"'
        in owner_function_source
    )
    assert (
        '"realigned_dynamic_boundary_shape_signature_map"'
        in owner_function_source
    )
    assert "model_ir.operators" not in owner_function_source
    assert "lower_from_onnx2tf" not in owner_source


def test_lowerer_final_shape_activation_convergence_reuses_one_index() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    helper_name = "_run_indexed_final_shape_activation_convergence"
    helper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert [call.func.id for call in calls] == [
        "ModelIRGraphIndex",
        "_run_indexed_shape_convergence_cleanup",
        "_sanitize_hardswish_tensor_shapes",
        "_reconcile_static_tensor_shapes",
        "_resolve_dynamic_reshape_shapes",
        "_reconcile_static_tensor_shapes",
        "_optimize_fuse_conv_activation_chains",
        "_reconcile_static_tensor_shapes",
    ]
    for call in calls[1:]:
        graph_index_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "graph_index"
        )
        assert isinstance(graph_index_keyword.value, ast.Name)
        assert graph_index_keyword.value.id == "graph_index"
    fusion_call = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_optimize_fuse_conv_activation_chains"
    )
    fusion_layout_keyword = next(
        keyword
        for keyword in fusion_call.keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(fusion_layout_keyword.value, ast.Name)
    assert fusion_layout_keyword.value.id == "layout_state"

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    invocation = lowerer.body[invocation_index].value
    assert len(invocation.args) == 1
    assert isinstance(invocation.args[0], ast.Name)
    assert invocation.args[0].id == "model_ir"
    assert lowerer.body[invocation_index - 1].value.func.id == (
        "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains"
    )
    assert lowerer.body[invocation_index + 1].value.func.id == (
        "run_boundary_input_normalization_cleanup"
    )

    fusion_wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_optimize_fuse_conv_activation_chains"
    )
    wrapper_calls = [
        node
        for node in ast.walk(fusion_wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_optimize_fuse_activation_chains_pass"
    ]
    assert len(wrapper_calls) == 1
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "activation_fusion.py"
    )
    owner_tree = ast.parse(owner_path.read_text(encoding="utf-8"))
    fusion = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "optimize_fuse_activation_chains"
    )
    fusion_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(fusion)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in fusion_call_names
    assert "operator_indices_for_normalized_types" in fusion_call_names
    assert "consumer_indices" in fusion_call_names
    assert "remove_operator" in fusion_call_names
    assert "_prune_unused_tensors" in fusion_call_names

    direct_production_calls = [
        statement.value
        for statement in lowerer.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_optimize_fuse_conv_activation_chains"
    ]
    assert len(direct_production_calls) == 2
    for call in direct_production_calls:
        layout_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_rank4_broadcast_constant_repair_uses_one_graph_index() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "binary_layout_adapter.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    repair = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "repair_rank4_channelwise_broadcast_constants_to_runtime_layout"
    )
    repair_source = ast.get_source_segment(owner_source, repair)
    assert repair_source is not None
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(repair)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in call_names
    assert "_build_tensor_producer_map" not in call_names
    assert "ModelIRGraphIndex" in call_names
    assert "operator_indices_for_types" in call_names
    assert "producer" in call_names
    assert "_set_operator_inputs" in call_names
    assert "for tensor_name, indices in graph_index.consumers.items()" in (
        repair_source
    )
    setter_call = next(
        node
        for node in ast.walk(repair)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_operator_inputs"
    )
    graph_index_keyword = next(
        keyword
        for keyword in setter_call.keywords
        if keyword.arg == "graph_index"
    )
    assert isinstance(graph_index_keyword.value, ast.Name)
    assert graph_index_keyword.value.id == "graph_index"


def test_stale_binary_layout_convergence_uses_one_graph_index() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "stale_binary_adapter_repair.py"
    )
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    repair_name = "_repair_stale_nchw_to_nhwc_channelwise_binary_transposes"
    repair = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == repair_name
    )
    assert "lower_from_onnx2tf" not in owner_source
    repair_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(repair)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in repair_call_names
    assert "_build_tensor_producer_map" not in repair_call_names
    assert "ModelIRGraphIndex" in repair_call_names
    assert "operator_indices_for_types" in repair_call_names
    assert "producer" in repair_call_names
    assert "consumer_indices" in repair_call_names
    assert "remove_operator" in repair_call_names
    channelwise_assignment = next(
        node
        for node in ast.walk(repair)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "channelwise_const_matches"
            for target in node.targets
        )
    )
    rank_guard = next(
        node
        for node in ast.walk(repair)
        if isinstance(node, ast.If)
        and "len(source_shape)" in ast.unparse(node.test)
        and "len(adapter_shape)" in ast.unparse(node.test)
    )
    assert rank_guard.lineno < channelwise_assignment.lineno
    setter_call = next(
        node
        for node in ast.walk(repair)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_operator_inputs"
    )
    source_signature_assignment = next(
        node
        for node in ast.walk(repair)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "source_signature"
            for target in node.targets
        )
    )
    assert source_signature_assignment.lineno < setter_call.lineno
    setter_index_keyword = next(
        keyword
        for keyword in setter_call.keywords
        if keyword.arg == "graph_index"
    )
    assert isinstance(setter_index_keyword.value, ast.Name)
    assert setter_index_keyword.value.id == "graph_index"

    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == repair_name
    )
    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == f"{repair_name}_pass"
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].args[0], ast.Name)
    assert wrapper_calls[0].args[0].id == "model_ir"
    wrapper_index_keyword = next(
        keyword
        for keyword in wrapper_calls[0].keywords
        if keyword.arg == "graph_index"
    )
    assert isinstance(wrapper_index_keyword.value, ast.Name)
    assert wrapper_index_keyword.value.id == "graph_index"

    helper_name = "_run_indexed_binary_layout_convergence"
    helper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    loop = next(node for node in helper.body if isinstance(node, ast.For))
    assert isinstance(loop.iter, ast.Call)
    assert isinstance(loop.iter.func, ast.Name)
    assert loop.iter.func.id == "range"
    assert len(loop.iter.args) == 1
    assert isinstance(loop.iter.args[0], ast.Constant)
    assert loop.iter.args[0].value == 3
    expected_order = [
        "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout",
        repair_name,
        "_reconcile_static_tensor_shapes",
    ]
    helper_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    ]
    assert [
        call.func.id
        for call in sorted(helper_calls, key=lambda candidate: candidate.lineno)
    ] == expected_order
    for call in helper_calls:
        index_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "graph_index"
        )
        assert isinstance(index_keyword.value, ast.Name)
        assert index_keyword.value.id == "graph_index"

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(invocations) == 2
    assert [
        call.args[0].id
        for call in sorted(invocations, key=lambda candidate: candidate.lineno)
    ] == [
        "fallback_ir",
        "model_ir",
    ]

    direct_repair_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == repair_name
    ]
    assert [
        call.args[0].id
        for call in sorted(
            direct_repair_invocations,
            key=lambda candidate: candidate.lineno,
        )
    ] == [
        "fallback_ir",
        "model_ir",
    ]


def test_conv_input_adapter_repairs_use_one_graph_index() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv_input_adapter_repair.py"
    )
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    assert "lower_from_onnx2tf" not in owner_source
    repair_names = [
        "_repair_singleton_nhwc_conv_input_reshapes",
        "_repair_stale_nchw_to_nhwc_conv_input_transposes",
    ]
    for repair_name in repair_names:
        repair = next(
            node
            for node in owner_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == repair_name
        )
        call_names = {
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            for node in ast.walk(repair)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert "_build_tensor_consumer_map" not in call_names
        assert "_build_tensor_producer_map" not in call_names
        assert "ModelIRGraphIndex" in call_names
        assert "operator_indices" in call_names
        assert "producer" in call_names
        assert "consumer_indices" in call_names
        assert "remove_operator" in call_names
        setter_call = next(
            node
            for node in ast.walk(repair)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_set_operator_inputs"
        )
        source_signature_assignment = next(
            node
            for node in ast.walk(repair)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "source_signature"
                for target in node.targets
            )
        )
        assert source_signature_assignment.lineno < setter_call.lineno
        setter_index_keyword = next(
            keyword
            for keyword in setter_call.keywords
            if keyword.arg == "graph_index"
        )
        assert isinstance(setter_index_keyword.value, ast.Name)
        assert setter_index_keyword.value.id == "graph_index"

        wrapper = next(
            node
            for node in lowering_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == repair_name
        )
        wrapper_calls = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == f"{repair_name}_pass"
        ]
        assert len(wrapper_calls) == 1
        wrapper_index_keyword = next(
            keyword
            for keyword in wrapper_calls[0].keywords
            if keyword.arg == "graph_index"
        )
        assert isinstance(wrapper_index_keyword.value, ast.Name)
        assert wrapper_index_keyword.value.id == "graph_index"

    helper_name = "_run_indexed_conv_input_adapter_repairs"
    helper = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    helper_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in repair_names
    ]
    assert [
        call.func.id
        for call in sorted(helper_calls, key=lambda candidate: candidate.lineno)
    ] == repair_names
    for call in helper_calls:
        index_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "graph_index"
        )
        assert isinstance(index_keyword.value, ast.Name)
        assert index_keyword.value.id == "graph_index"

    helper_wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    helper_wrapper_calls = [
        node
        for node in ast.walk(helper_wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == f"{helper_name}_pass"
    ]
    assert len(helper_wrapper_calls) == 1

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert [
        call.args[0].id
        for call in sorted(
            helper_invocations,
            key=lambda candidate: candidate.lineno,
        )
    ] == ["model_ir", "fallback_ir"]
    direct_transpose_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == repair_names[1]
    ]
    assert len(direct_transpose_invocations) == 1
    assert direct_transpose_invocations[0].args[0].id == "model_ir"


def test_mixed_nhwc_nchw_concat_repair_has_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "mixed_concat_input_repair.py"
    )
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    assert "lower_from_onnx2tf" not in owner_source
    owner_name = "_repair_mixed_nhwc_inputs_for_nchw_concat"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_clone_quantization" in owner_calls
    assert "_quant_scale_count" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "insert" in owner_calls

    tensor_insertions = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "model_ir"
            and target.value.attr == "tensors"
            for target in node.targets
        )
    ]
    operator_insertions = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
    ]
    input_setters = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_operator_inputs"
    ]
    first_mutation_line = min(
        node.lineno
        for node in tensor_insertions + operator_insertions + input_setters
    )
    output_assignment = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "output_tensor"
            for target in node.targets
        )
    )
    source_signature_assignment = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "source_signature"
            for target in node.targets
        )
    )
    plan_append = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "adapter_plans"
        and node.func.attr == "append"
    )
    assert output_assignment.lineno < first_mutation_line
    assert source_signature_assignment.lineno < first_mutation_line
    assert plan_append.lineno < first_mutation_line

    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == f"{owner_name}_pass"
    ]
    assert len(wrapper_calls) == 1

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
    ]
    assert [
        call.args[0].id
        for call in sorted(invocations, key=lambda candidate: candidate.lineno)
    ] == ["fallback_ir", "model_ir"]


def test_wrong_way_conv_transpose_sanitizer_has_one_indexed_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv_input_layout.py"
    )
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    swish_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_swish_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    swish_tree = ast.parse(swish_path.read_text(encoding="utf-8"))
    owner_name = "sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    wrapper_name = "_sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv"
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in call_names
    assert "ModelIRGraphIndex" in call_names
    assert "operator_indices" in call_names
    assert "consumer_indices" in call_names
    assert "_replace_tensor_inputs" in call_names
    assert "remove_operator" in call_names
    replacement_call = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_replace_tensor_inputs"
    )
    index_keyword = next(
        keyword
        for keyword in replacement_call.keywords
        if keyword.arg == "graph_index"
    )
    assert isinstance(index_keyword.value, ast.Name)
    assert index_keyword.value.id == "active_index"

    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv_pass"
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].args[0], ast.Name)
    assert wrapper_calls[0].args[0].id == "model_ir"

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(invocations) == 1
    assert isinstance(invocations[0].args[0], ast.Name)
    assert invocations[0].args[0].id == "model_ir"

    swish_owner = next(
        node
        for node in swish_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "optimize_transpose_swish_qdq_nhwc_islands"
    )
    swish_invocations = [
        node
        for node in ast.walk(swish_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
    ]
    assert len(swish_invocations) == 1


def test_quantized_swish_primary_phase_has_one_indexed_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_swish_layout.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_tree = ast.parse(owner_path.read_text(encoding="utf-8"))
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    branch_owner_name = "rewrite_transpose_swish_qdq_nhwc_branches"
    branch_owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == branch_owner_name
    )
    branch_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(branch_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in branch_calls
    assert "_build_tensor_producer_map" not in branch_calls
    assert "ModelIRGraphIndex" in branch_calls
    assert "operator_indices" in branch_calls
    assert "consumer_indices" in branch_calls
    assert "_set_operator_inputs" in branch_calls
    assert "remove_operator" in branch_calls
    setter_calls = [
        node
        for node in ast.walk(branch_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_operator_inputs"
    ]
    assert len(setter_calls) == 2
    assert all(
        any(
            keyword.arg == "graph_index"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "active_index"
            for keyword in call.keywords
        )
        for call in setter_calls
    )

    metadata_owner_name = "propagate_swish_qdq_nhwc_metadata"
    metadata_owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == metadata_owner_name
    )
    metadata_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(metadata_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in metadata_calls
    assert "_build_tensor_producer_map" not in metadata_calls
    assert "ModelIRGraphIndex" in metadata_calls
    assert "operator_indices_for_types" in metadata_calls
    shape_owner_name = "copy_swish_qdq_shape_signature"
    assert shape_owner_name in metadata_calls

    runner_name = "run_swish_qdq_nhwc_primary_phases"
    runner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == runner_name
    )
    runner_index_builds = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRGraphIndex"
    ]
    assert len(runner_index_builds) == 1
    runner_phase_calls = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {branch_owner_name, metadata_owner_name}
    ]
    assert [call.func.id for call in runner_phase_calls] == [
        branch_owner_name,
        metadata_owner_name,
    ]
    assert all(
        any(
            keyword.arg == "graph_index"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "graph_index"
            for keyword in call.keywords
        )
        for call in runner_phase_calls
    )

    orchestrator_name = "optimize_transpose_swish_qdq_nhwc_islands"
    swish_orchestrator = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == orchestrator_name
    )
    runner_invocations = [
        node
        for node in ast.walk(swish_orchestrator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == runner_name
    ]
    assert len(runner_invocations) == 1
    assert isinstance(runner_invocations[0].args[0], ast.Name)
    assert runner_invocations[0].args[0].id == "model_ir"
    post_owner_name = "remove_inverse_post_transposes_for_swish_qdq"
    post_owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == post_owner_name
    )
    post_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(post_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in post_calls
    assert "_build_tensor_producer_map" not in post_calls
    assert "ModelIRGraphIndex" in post_calls
    assert "operator_indices" in post_calls
    assert "_replace_tensor_inputs" in post_calls
    assert "remove_operator" in post_calls

    late_concat_owner_name = "normalize_late_swish_qdq_concat_inputs"
    late_concat_owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == late_concat_owner_name
    )
    late_concat_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(late_concat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in late_concat_calls
    assert "_build_tensor_producer_map" not in late_concat_calls
    assert "ModelIRGraphIndex" in late_concat_calls
    assert "operator_indices" in late_concat_calls
    assert "producer" in late_concat_calls
    assert "consumer_indices" in late_concat_calls
    assert "_set_operator_inputs" in late_concat_calls
    assert "remove_operators" in late_concat_calls
    assert shape_owner_name in late_concat_calls

    late_runner_name = "run_swish_qdq_late_concat_and_post_cleanup"
    late_runner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == late_runner_name
    )
    late_runner_index_builds = [
        node
        for node in ast.walk(late_runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRGraphIndex"
    ]
    assert len(late_runner_index_builds) == 1
    late_phase_calls = [
        node
        for node in ast.walk(late_runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {late_concat_owner_name, post_owner_name}
    ]
    assert [call.func.id for call in late_phase_calls] == [
        late_concat_owner_name,
        post_owner_name,
    ]
    assert all(
        any(
            keyword.arg == "graph_index"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "graph_index"
            for keyword in call.keywords
        )
        for call in late_phase_calls
    )

    post_invocations = sorted(
        [
            node
            for node in ast.walk(swish_orchestrator)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == post_owner_name
        ],
        key=lambda node: node.lineno,
    )
    assert len(post_invocations) == 1
    assert all(
        len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "model_ir"
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "rewritten_tensors"
        for call in post_invocations
    )
    late_runner_invocations = [
        node
        for node in ast.walk(swish_orchestrator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == late_runner_name
    ]
    assert len(late_runner_invocations) == 1
    assert isinstance(late_runner_invocations[0].args[0], ast.Name)
    assert late_runner_invocations[0].args[0].id == "model_ir"
    assert post_invocations[0].lineno < late_runner_invocations[0].lineno
    assert not any(
        isinstance(node, ast.While) for node in swish_orchestrator.body
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == shape_owner_name
        for node in ast.walk(swish_orchestrator)
    )
    nested_names = {
        node.name
        for node in swish_orchestrator.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_is_swish_quantized_output" not in nested_names
    assert "_concat_has_quantize_transpose_tail" not in nested_names
    assert "_has_concat_closure_from_tensor" not in nested_names
    assert "_copy_shape_signature" not in nested_names

    orchestrator_calls = [
        node.func.id
        for node in ast.walk(swish_orchestrator)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert orchestrator_calls.count(
        "sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv"
    ) == 1
    assert orchestrator_calls.count("_prune_unused_tensors") == 1
    assert "lower_from_onnx2tf" not in owner_path.read_text(encoding="utf-8")

    wrapper_name = "_optimize_transpose_swish_qdq_nhwc_islands"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == f"_{orchestrator_name}_pass"
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].args[0], ast.Name)
    assert wrapper_calls[0].args[0].id == "model_ir"
    assert {
        keyword.arg for keyword in wrapper_calls[0].keywords
    } == {"min_spatial_stage", "require_concat_closure"}

    closure_owner_name = (
        "optimize_transpose_swish_residual_concat_closure_nhwc_chains"
    )
    closure_owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == closure_owner_name
    )
    closure_owner_calls = [
        node
        for node in ast.walk(closure_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == orchestrator_name
    ]
    assert len(closure_owner_calls) == 1
    closure_keywords = {
        keyword.arg: keyword.value for keyword in closure_owner_calls[0].keywords
    }
    assert isinstance(closure_keywords["min_spatial_stage"], ast.Constant)
    assert closure_keywords["min_spatial_stage"].value == 0
    assert isinstance(closure_keywords["require_concat_closure"], ast.Constant)
    assert closure_keywords["require_concat_closure"].value is True

    closure_wrapper_name = (
        "_optimize_transpose_swish_residual_concat_closure_nhwc_chains"
    )
    closure_wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == closure_wrapper_name
    )
    closure_wrapper_calls = [
        node
        for node in ast.walk(closure_wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == f"_{closure_owner_name}_pass"
    ]
    assert len(closure_wrapper_calls) == 1

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node.func.id
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {wrapper_name, closure_wrapper_name}
    ]
    assert production_calls.count(wrapper_name) == 1
    assert production_calls.count(closure_wrapper_name) == 1


def test_hardswish_se_layout_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "hardswish_se_layout.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_tree = ast.parse(lowerer_source)
    owner_name = (
        "optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
    )
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_calls = {
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_tensor_consumer_map" in owner_calls
    assert "_read_transpose_perm" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "_prune_unused_tensors" in owner_calls
    assert "lower_from_onnx2tf" not in owner_source

    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == f"_{owner_name}_pass"
    ]
    assert len(dispatches) == 1
    assert isinstance(dispatches[0].args[0], ast.Name)
    assert dispatches[0].args[0].id == "model_ir"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2


def test_nhwc_concat_legacy_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "nhwc_concat_legacy_layout.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_tree = ast.parse(lowerer_source)
    owner_name = "optimize_transpose_pre_concat_nhwc_chains_legacy"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_calls = {
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_tensor_consumer_map" in owner_calls
    assert "_build_tensor_producer_map" in owner_calls
    assert "_read_transpose_perm" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "_prune_unused_tensors" in owner_calls
    assert "lower_from_onnx2tf" not in owner_source

    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    dispatch = wrapper.body[0].value
    assert isinstance(dispatch, ast.Call)
    assert isinstance(dispatch.func, ast.Name)
    assert dispatch.func.id == f"_{owner_name}_pass"
    assert len(dispatch.args) == 1
    assert isinstance(dispatch.args[0], ast.Name)
    assert dispatch.args[0].id == "model_ir"

    composite_name = "_optimize_transpose_pre_concat_nhwc_chains"
    composite = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == composite_name
    )
    expected_dispatch_order = [
        "run_nhwc_concat_layout_cleanup",
        "run_nhwc_concat_quantized_layout_cleanup",
        wrapper_name,
    ]
    composite_dispatches = [
        node
        for node in ast.walk(composite)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_dispatch_order
    ]
    assert len(composite_dispatches) == len(expected_dispatch_order)
    assert [
        node.func.id
        for node in sorted(composite_dispatches, key=lambda node: node.lineno)
    ] == expected_dispatch_order

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == composite_name
    ]
    assert len(production_calls) == 4


def test_slice_prepost_layout_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "slice_prepost_layout.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    owner_name = "optimize_transpose_slice_prepost_nhwc_passthrough_chains"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_calls = {
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_tensor_consumer_map" in owner_calls
    assert "_build_tensor_producer_map" in owner_calls
    assert "_infer_slice_output_shape_and_resolved_params" in owner_calls
    assert "_read_transpose_perm" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "_prune_unused_tensors" in owner_calls
    assert "lower_from_onnx2tf" not in owner_source

    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    dispatch = wrapper.body[0].value
    assert isinstance(dispatch, ast.Call)
    assert isinstance(dispatch.func, ast.Name)
    assert dispatch.func.id == f"_{owner_name}_pass"
    assert len(dispatch.args) == 1
    assert isinstance(dispatch.args[0], ast.Name)
    assert dispatch.args[0].id == "model_ir"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1


def test_shape_extract_layout_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "shape_extract_layout.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    owner_name = "optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_calls = {
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_tensor_consumer_map" in owner_calls
    assert "_clone_quantization" in owner_calls
    assert "_read_transpose_perm" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_write_const_ints_to_tensor" in owner_calls
    assert "_prune_unused_tensors" in owner_calls
    assert "lower_from_onnx2tf" not in owner_source

    wrapper_name = f"_{owner_name}"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    dispatch = wrapper.body[0].value
    assert isinstance(dispatch, ast.Call)
    assert isinstance(dispatch.func, ast.Name)
    assert dispatch.func.id == f"_{owner_name}_pass"
    assert len(dispatch.args) == 1
    assert isinstance(dispatch.args[0], ast.Name)
    assert dispatch.args[0].id == "model_ir"

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3


def test_recurrent_alias_repair_has_one_shared_indexed_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "recurrent_alias.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pytorch_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pytorch_recurrent.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "repair_orphan_recurrent_step_tensors"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_producer_map" not in owner_call_names
    assert "_build_tensor_consumer_map" not in owner_call_names
    assert "ModelIRGraphIndex" in owner_call_names
    assert "producer" in owner_call_names
    assert "consumers_of" in owner_call_names
    assert "operator_index" in owner_call_names
    assert "replace_operator_inputs" in owner_call_names
    assert "for op in model_ir.operators" not in owner_source

    wrapper_name = "_repair_orphan_recurrent_step_tensors"
    for wrapper_path in [lowerer_path, pytorch_path]:
        wrapper_source = wrapper_path.read_text(encoding="utf-8")
        wrapper_tree = ast.parse(wrapper_source)
        assert f"def {owner_name}(" not in wrapper_source
        wrapper = next(
            node
            for node in wrapper_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        owner_calls = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner_name
        ]
        assert len(owner_calls) == 1
        index_keyword = next(
            keyword
            for keyword in owner_calls[0].keywords
            if keyword.arg == "graph_index"
        )
        assert isinstance(index_keyword.value, ast.Name)
        assert index_keyword.value.id == "graph_index"


def test_unbound_input_layout_repair_has_one_indexed_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "unbound_input_layout.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "repair_unbound_nonconstant_inputs_with_layout_transpose"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_producer_map" not in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "model_ir.operators.insert" not in owner_source
    assert "while True" not in owner_source
    assert owner_source.count("ModelIRGraphIndex(model_ir)") == 1
    assert "producer" in owner_call_names
    assert "operator_index" in owner_call_names
    assert "consumer_indices" in owner_call_names
    assert "insert_operator" in owner_source

    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)
    wrappers = {
        node.name: node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_find_unbound_nonconstant_operator_inputs",
            "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose",
        }
    }
    assert set(wrappers) == {
        "_find_unbound_nonconstant_operator_inputs",
        "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose",
    }
    find_source = ast.get_source_segment(
        lowerer_source,
        wrappers["_find_unbound_nonconstant_operator_inputs"],
    )
    repair_source = ast.get_source_segment(
        lowerer_source,
        wrappers[
            "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
        ],
    )
    assert find_source is not None
    assert repair_source is not None
    assert "find_unbound_nonconstant_operator_inputs(" in find_source
    assert owner_name in repair_source
    assert "graph_index=result.graph_index" in repair_source
    assert "_build_tensor_producer_map" not in repair_source
    assert "_build_tensor_consumer_map" not in repair_source

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    wrapper_name = (
        "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
    )
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert [
        call.args[0].id
        for call in sorted(invocations, key=lambda candidate: candidate.lineno)
    ] == ["model_ir", "fallback_ir"]


def test_quantized_activation_bridge_cleanup_has_one_indexed_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_activation.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "optimize_transpose_dequant_relu_quantize_bridges"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_source
    assert "model_ir.operators.remove" not in owner_source
    assert "ModelIRGraphIndex" in call_names
    assert "operator_indices" in call_names
    assert "consumer_indices" in call_names
    assert "remove_operators" in call_names
    for setter_name in ["_set_operator_inputs", "_set_operator_outputs"]:
        setter_call = next(
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == setter_name
        )
        index_keyword = next(
            keyword
            for keyword in setter_call.keywords
            if keyword.arg == "graph_index"
        )
        assert isinstance(index_keyword.value, ast.Name)
        assert index_keyword.value.id == "active_index"

    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)
    wrapper_name = "_optimize_transpose_dequant_relu_quantize_bridges"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
    ]
    assert len(wrapper_calls) == 1
    assert "_build_tensor_consumer_map" not in ast.get_source_segment(
        lowerer_source,
        wrapper,
    )


def test_complex_quantized_activation_bridges_have_one_indexed_owner() -> None:
    owner_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "quantized_activation.py"
    )
    lowerer_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)
    owners = [
        (
            "optimize_transpose_dequant_hardsigmoid_quantize_bridges",
            "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
        ),
        (
            "optimize_transpose_dequant_mul_add_prelu_quantize_bridges",
            "_optimize_transpose_dequant_mul_add_prelu_quantize_bridges",
        ),
    ]

    for owner_name, wrapper_name in owners:
        owner = next(
            node
            for node in owner_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == owner_name
        )
        owner_segment = ast.get_source_segment(owner_source, owner)
        assert owner_segment is not None
        call_names = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert "_build_tensor_consumer_map" not in owner_segment
        assert "del model_ir.operators" not in owner_segment
        for call_name in [
            "ModelIRGraphIndex",
            "operator_indices",
            "consumer_indices",
            "remove_operators",
            "_plan_constant_layout_remaps",
            "_apply_constant_layout_remaps",
        ]:
            assert call_name in call_names
        for setter_name in ["_set_operator_inputs", "_set_operator_outputs"]:
            setter_call = next(
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == setter_name
            )
            index_keyword = next(
                keyword
                for keyword in setter_call.keywords
                if keyword.arg == "graph_index"
            )
            assert isinstance(index_keyword.value, ast.Name)
            assert index_keyword.value.id == "active_index"

        wrapper = next(
            node
            for node in lowerer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        wrapper_calls = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner_name
        ]
        assert len(wrapper_calls) == 1
        assert "_build_tensor_consumer_map" not in ast.get_source_segment(
            lowerer_source,
            wrapper,
        )

    apply_helper = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_constant_layout_remaps"
    )
    replacement_call = next(
        node
        for node in ast.walk(apply_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_replace_operator_input_at"
    )
    index_keyword = next(
        keyword
        for keyword in replacement_call.keywords
        if keyword.arg == "graph_index"
    )
    assert isinstance(index_keyword.value, ast.Name)
    assert index_keyword.value.id == "graph_index"


def test_quantized_logistic_gate_bridge_has_one_indexed_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_gate.py"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_name = "optimize_transpose_dequant_logistic_mul_quantize_bridges"
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == owner_name
    )
    owner_segment = ast.get_source_segment(owner_source, owner)
    assert owner_segment is not None
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_segment
    assert "_build_tensor_producer_map" not in owner_segment
    assert "del model_ir.operators" not in owner_segment
    for call_name in [
        "ModelIRGraphIndex",
        "operator_indices",
        "consumer_indices",
        "remove_operators",
    ]:
        assert call_name in call_names
    for setter_name in [
        "_set_operator_inputs",
        "_set_operator_outputs",
        "_replace_tensor_inputs",
    ]:
        setter_calls = [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == setter_name
        ]
        assert setter_calls
        for setter_call in setter_calls:
            index_keyword = next(
                keyword
                for keyword in setter_call.keywords
                if keyword.arg == "graph_index"
            )
            assert isinstance(index_keyword.value, ast.Name)
            assert index_keyword.value.id == "active_index"

    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)
    wrapper_name = "_optimize_transpose_dequant_logistic_mul_quantize_bridges"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
    ]
    assert len(wrapper_calls) == 1
    wrapper_segment = ast.get_source_segment(lowerer_source, wrapper)
    assert wrapper_segment is not None
    assert "_build_tensor_consumer_map" not in wrapper_segment
    assert "_build_tensor_producer_map" not in wrapper_segment


def test_lowerer_late_layout_qkv_bridge_pair_stays_between_raw_rewrites() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_qkv_attention_layout_pass_cluster"
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
        and any(
            keyword.arg == "include_prefix"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in statement.value.keywords
        )
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )


def test_lowerer_duplicate_quantized_prelu_pair_reuses_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_duplicate_quantized_prelu_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_duplicate_fanout_cleanup",
        "run_quantized_prelu_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"
    duplicate_include_transpose = next(
        keyword
        for keyword in calls["run_duplicate_fanout_cleanup"].keywords
        if keyword.arg == "include_transpose"
    )
    assert isinstance(duplicate_include_transpose.value, ast.Name)
    assert duplicate_include_transpose.value.id == "include_transpose"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 1
    for call in helper_invocations:
        include_transpose = next(
            keyword
            for keyword in call.keywords
            if keyword.arg == "include_transpose"
        )
        assert isinstance(include_transpose.value, ast.Name)
        assert (
            include_transpose.value.id
            == "include_duplicate_transpose"
        )


def test_lowerer_very_late_gather_constant_normalization_cluster_reuses_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_very_late_gather_constant_normalization_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_transpose_gather_axis_cleanup",
        "_run_constant_fold_cast_cleanup_pass_cluster",
        "run_normalization_pad_layout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }
    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"
    normalization_call = calls["run_normalization_pad_layout_cleanup"]
    include_instance = next(
        keyword
        for keyword in normalization_call.keywords
        if keyword.arg == "include_instance"
    )
    include_flatten = next(
        keyword
        for keyword in normalization_call.keywords
        if keyword.arg == "include_flatten"
    )
    assert isinstance(include_instance.value, ast.Constant)
    assert include_instance.value.value is False
    assert isinstance(include_flatten.value, ast.Constant)
    assert include_flatten.value.value is True

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert next_boundary.value.func.id == "_resolve_dynamic_reshape_shapes"


def test_lowerer_constant_fold_cast_pair_reuses_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_constant_fold_cast_cleanup_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_constant_input_fold_cleanup",
        "run_redundant_cast_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 2
    external_scope_invocations = [
        call
        for call in helper_invocations
        if any(keyword.arg == "state_scope" for keyword in call.keywords)
    ]
    assert len(external_scope_invocations) == 2
    for call in external_scope_invocations:
        scope_keyword = next(
            keyword
            for keyword in call.keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"


def test_lowerer_se_fc_gather_fanout_pair_reuses_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_se_fc_gather_channel_fanout_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_se_fc_layout_cleanup",
        "run_transpose_gather_channel_fanout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"
    assert [argument.arg for argument in helper.args.args] == [
        "target_model_ir",
        "target_layout_state",
    ]

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 2
    assert sum(
        len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "fallback_ir"
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value is None
        for call in helper_invocations
    ) == 1
    assert sum(
        len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "model_ir"
        and isinstance(call.args[1], ast.Attribute)
        and call.args[1].attr == "layout_state"
        for call in helper_invocations
    ) == 1


def test_lowerer_terminal_boundary_layout_cluster_reuses_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_terminal_boundary_layout_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_dual_mul_concat_layout_cleanup",
        "run_boundary_input_layout_cleanup",
        "run_pad_layout_cleanup",
        "run_layout_transpose_cleanup",
        "run_transpose_gather_channel_fanout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.If)
    assert isinstance(next_boundary.test, ast.Name)
    assert next_boundary.test.id == "optimize_layout_transpose_chains"


def test_lowerer_late_dequant_unary_fanout_cluster_reuses_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_late_dequant_unary_fanout_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_dequant_concat_quantize_layout_cleanup",
        "run_transpose_unary_passthrough_cleanup",
        "run_transpose_unary_fanout_bridge_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_dequant_hardsigmoid_quantize_bridges"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_optimize_swish_transpose_passthrough_chains"
    )


def test_lowerer_terminal_singleton_maxpool_reshape_pair_reuses_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_terminal_singleton_maxpool_reshape_pass_pair"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_singleton_maxpool_layout_cleanup",
        "run_consecutive_reshape_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.If)
    assert isinstance(previous_boundary.test, ast.Name)
    assert previous_boundary.test.id == "optimize_layout_transpose_chains"
    previous_call = previous_boundary.body[0]
    assert isinstance(previous_call, ast.Expr)
    assert isinstance(previous_call.value, ast.Call)
    assert isinstance(previous_call.value.func, ast.Name)
    assert (
        previous_call.value.func.id
        == "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.If)
    assert isinstance(next_boundary.test, ast.Name)
    assert next_boundary.test.id == "optimize_layout_transpose_chains"
    next_call = next_boundary.body[0]
    assert isinstance(next_call, ast.Expr)
    assert isinstance(next_call.value, ast.Call)
    assert isinstance(next_call.value.func, ast.Name)
    assert (
        next_call.value.func.id
        == "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
    )


def test_lowerer_terminal_clamp_unary_relu_cluster_reuses_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_terminal_clamp_unary_relu_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_clamp_cleanup",
        "run_transpose_unary_passthrough_cleanup",
        "run_maximum_zero_relu_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.If)
    assert isinstance(previous_boundary.test, ast.Name)
    assert previous_boundary.test.id == "optimize_layout_transpose_chains"
    previous_call = previous_boundary.body[-1]
    assert isinstance(previous_call, ast.Expr)
    assert isinstance(previous_call.value, ast.Call)
    assert isinstance(previous_call.value.func, ast.Name)
    assert (
        previous_call.value.func.id
        == "_run_singleton_reshape_layout_pass_cluster"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_run_sinet_terminal_layout_recovery_sequence"
    )


def test_lowerer_late_layout_mean_spp_gather_constant_cast_cluster_reuses_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_late_layout_mean_spp_gather_constant_cast_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_layout_transpose_cleanup",
        "run_mean_mul_add_conv_layout_cleanup",
        "run_spp_layout_cleanup",
        "run_transpose_gather_axis_cleanup",
        "_run_constant_fold_cast_cleanup_pass_cluster",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    conditional = next(
        statement
        for statement in helper.body
        if isinstance(statement, ast.If)
    )
    assert isinstance(conditional.test, ast.Name)
    assert conditional.test.id == "include_layout_transpose"
    assert calls["run_layout_transpose_cleanup"] in [
        node for node in ast.walk(conditional)
    ]

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    invocation = lowerer.body[invocation_index].value
    include_layout = next(
        keyword
        for keyword in invocation.keywords
        if keyword.arg == "include_layout_transpose"
    )
    assert isinstance(include_layout.value, ast.Name)
    assert include_layout.value.id == "optimize_layout_transpose_chains"

    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_replace_expand_dims_and_squeeze_with_reshape"
    )


def test_lowerer_late_spp_concat_unary_conv_pair_reuses_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_late_spp_concat_unary_conv_pass_pair"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_spp_layout_cleanup",
        "run_concat_unary_conv_layout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )


def test_lowerer_late_hard_activation_layout_pair_reuses_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_late_hard_activation_layout_pass_pair"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_hard_activation_passthrough_cleanup",
        "run_layout_transpose_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }
    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    hard_activation_call = calls["run_hard_activation_passthrough_cleanup"]
    expected_flags = {
        "include_hardswish": False,
        "include_hardsigmoid": True,
        "include_hardsigmoid_mul": True,
        "reverse_hardsigmoid_order": True,
    }
    for name, expected_value in expected_flags.items():
        keyword = next(
            candidate
            for candidate in hard_activation_call.keywords
            if candidate.arg == name
        )
        assert isinstance(keyword.value, ast.Constant)
        assert keyword.value.value is expected_value

    conditional = next(
        statement
        for statement in helper.body
        if isinstance(statement, ast.If)
    )
    assert isinstance(conditional.test, ast.Name)
    assert conditional.test.id == "include_layout_transpose"
    assert calls["run_layout_transpose_cleanup"] in [
        node for node in ast.walk(conditional)
    ]

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    invocation = lowerer.body[invocation_index].value
    include_layout = next(
        keyword
        for keyword in invocation.keywords
        if keyword.arg == "include_layout_transpose"
    )
    assert isinstance(include_layout.value, ast.Name)
    assert include_layout.value.id == "optimize_layout_transpose_chains"

    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_optimize_transpose_pre_concat_nhwc_chains"
    )


def test_lowerer_absolute_final_normalization_attention_pair_reuses_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_absolute_final_normalization_attention_pass_pair"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_normalization_pad_layout_cleanup",
        "run_mixed_attention_layout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"
    normalization_call = calls["run_normalization_pad_layout_cleanup"]
    include_instance = next(
        keyword
        for keyword in normalization_call.keywords
        if keyword.arg == "include_instance"
    )
    include_flatten = next(
        keyword
        for keyword in normalization_call.keywords
        if keyword.arg == "include_flatten"
    )
    assert isinstance(include_instance.value, ast.Constant)
    assert include_instance.value.value is False
    assert isinstance(include_flatten.value, ast.Constant)
    assert include_flatten.value.value is True

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    )


def test_indexed_instance_norm_post_bias_owner_has_one_core_match_contract() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    core_source = (pass_root / "decomposed_instance_norm.py").read_text(
        encoding="utf-8"
    )
    prepost_source = (pass_root / "instance_norm_prepost_layout.py").read_text(
        encoding="utf-8"
    )
    post_bias_source = (pass_root / "instance_norm_post_bias_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )

    assert "def match_decomposed_instance_norm_core(" in core_source
    assert "match_decomposed_instance_norm_core(" in prepost_source
    assert "match_decomposed_instance_norm_core(" in post_bias_source
    assert "_build_tensor_consumer_map" not in post_bias_source
    assert "while True" not in post_bias_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }


def test_indexed_instance_norm_residual_add_owner_has_one_adapter_contract() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    common_source = (pass_root / "decomposed_instance_norm.py").read_text(
        encoding="utf-8"
    )
    post_bias_source = (
        pass_root / "instance_norm_post_bias_layout.py"
    ).read_text(encoding="utf-8")
    residual_source = (
        pass_root / "instance_norm_residual_add_layout.py"
    ).read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = (
        "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )

    assert "def plan_nhwc_instance_norm_constant_updates(" in common_source
    assert "plan_nhwc_instance_norm_constant_updates(" in post_bias_source
    assert "plan_nhwc_instance_norm_constant_updates(" in residual_source
    assert "match_decomposed_instance_norm_core(" in residual_source
    assert "_build_tensor_consumer_map" not in residual_source
    assert "_build_tensor_producer_map" not in residual_source
    assert "while True" not in residual_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    assert all(
        any(keyword.arg == "layout_state" for keyword in call.keywords)
        for call in production_calls
    )
    indexed_call = next(
        call
        for call in production_calls
        if any(keyword.arg == "graph_index" for keyword in call.keywords)
    )
    graph_keyword = next(
        keyword for keyword in indexed_call.keywords if keyword.arg == "graph_index"
    )
    assert isinstance(graph_keyword.value, ast.Name)
    assert graph_keyword.value.id == "residual_graph_index"


def test_indexed_instance_norm_residual_mul_concat_owner_is_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    common_source = (pass_root / "decomposed_instance_norm.py").read_text(
        encoding="utf-8"
    )
    owner_source = (
        pass_root / "instance_norm_residual_mul_concat_layout.py"
    ).read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = (
        "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )

    assert "additional_coefficient_uses" in common_source
    assert "plan_nhwc_instance_norm_constant_updates(" in owner_source
    assert "additional_coefficient_uses=" in owner_source
    assert "match_decomposed_instance_norm_core(" in owner_source
    assert "Counter(concat_inputs)" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 4
    assert all(
        any(keyword.arg == "layout_state" for keyword in call.keywords)
        for call in production_calls
    )
    indexed_calls = [
        call
        for call in production_calls
        if any(keyword.arg == "graph_index" for keyword in call.keywords)
    ]
    assert len(indexed_calls) == 1
    graph_keyword = next(
        keyword
        for keyword in indexed_calls[0].keywords
        if keyword.arg == "graph_index"
    )
    assert isinstance(graph_keyword.value, ast.Name)
    assert graph_keyword.value.id == "residual_graph_index"


def test_indexed_instance_norm_dual_stats_owner_keeps_distinct_path_contract() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    common_source = (pass_root / "decomposed_instance_norm.py").read_text(
        encoding="utf-8"
    )
    owner_source = (pass_root / "instance_norm_dual_stats_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )

    assert "def plan_nhwc_coefficient_updates(" in common_source
    assert "def _match_path(" in owner_source
    assert "plan_nhwc_coefficient_updates(" in owner_source
    assert "match_decomposed_instance_norm_core(" not in owner_source
    assert "tail_add_contract" in owner_source
    assert "Counter(centered_users)" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 4
    assert all(
        any(keyword.arg == "layout_state" for keyword in call.keywords)
        for call in production_calls
    )
    indexed_calls = [
        call
        for call in production_calls
        if any(keyword.arg == "graph_index" for keyword in call.keywords)
    ]
    assert len(indexed_calls) == 1
    graph_keyword = next(
        keyword
        for keyword in indexed_calls[0].keywords
        if keyword.arg == "graph_index"
    )
    assert isinstance(graph_keyword.value, ast.Name)
    assert graph_keyword.value.id == "residual_graph_index"


def test_indexed_affine_chain_fold_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "affine_chain_fold.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = "_optimize_fold_mul_add_mul_affine_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "Counter(" in owner_source
    assert "np.broadcast_shapes(" in owner_source
    assert "fusedActivationFunction" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "_optimize_fold_mul_add_mul_affine_chains_pass"
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3
    assert all(
        any(keyword.arg == "layout_state" for keyword in call.keywords)
        for call in production_calls
    )


def test_indexed_affine_prepost_layout_owner_preserves_canonical_output() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "affine_prepost_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )

    assert "def _constant_replacement(" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "canonical_output_name" in owner_source
    assert "Counter(" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert "if False" not in owner_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_transpose_mul_add_const_prepost_nhwc_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 7
    assert all(
        any(keyword.arg == "layout_state" for keyword in call.keywords)
        for call in production_calls
    )


def test_indexed_affine_post_add_layout_owner_is_bounded_and_separate_from_pad() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "affine_post_add_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )

    assert "passes.affine_prepost_layout import" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "output_shape=post_output.shape" in owner_source
    assert "output_signature=post_output.signature" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_transpose_mul_posttranspose_add_nhwc_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 4
    assert all(
        any(keyword.arg == "layout_state" for keyword in call.keywords)
        for call in production_calls
    )

    pad_wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_transpose_pad_mul_posttranspose_add_nhwc_chains"
    )
    pad_names = {
        node.id for node in ast.walk(pad_wrapper) if isinstance(node, ast.Name)
    }
    assert (
        "_optimize_transpose_pad_mul_posttranspose_add_nhwc_chains_pass"
        in pad_names
    )
    assert (
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains_pass"
        not in pad_names
    )


def test_indexed_sinet_shuffle_residual_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_shuffle_residual_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper_name = "_optimize_sinet_shuffle_residual_transpose_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _resolve_prefix(" in owner_source
    assert "def _resolve_postmul_candidate(" in owner_source
    assert "def _apply_postmul_plan(" in owner_source
    assert "def _resolve_late_candidate(" in owner_source
    assert "def _apply_late_plan(" in owner_source
    assert "def _late_constant_replacement(" in owner_source
    assert "def _plan_constants(" in owner_source
    assert "Counter(" in owner_source
    assert "metadata_updates" in owner_source
    assert "post1_tensor" in owner_source
    assert "post2_tensor" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_shuffle_residual_transpose_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"

    postmul_wrapper_name = (
        "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains"
    )
    postmul_wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == postmul_wrapper_name
    )
    assert len(postmul_wrapper.body) == 2
    postmul_dispatch = postmul_wrapper.body[1]
    assert isinstance(postmul_dispatch, ast.Return)
    postmul_call = next(
        node for node in ast.walk(postmul_dispatch) if isinstance(node, ast.Call)
    )
    assert isinstance(postmul_call.func, ast.Name)
    assert (
        postmul_call.func.id
        == "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains_pass"
    )
    assert {keyword.arg for keyword in postmul_call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    postmul_production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == postmul_wrapper_name
    ]
    assert len(postmul_production_calls) == 2
    postmul_layout_values = [
        next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "layout_state"
        )
        for call in postmul_production_calls
    ]
    assert any(
        isinstance(value, ast.Constant) and value.value is None
        for value in postmul_layout_values
    )
    assert any(
        isinstance(value, ast.Attribute) and value.attr == "layout_state"
        for value in postmul_layout_values
    )

    late_wrapper_name = (
        "_optimize_sinet_late_residual_pre_add_mul_add_prelu_chains"
    )
    late_wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == late_wrapper_name
    )
    assert len(late_wrapper.body) == 2
    late_dispatch = late_wrapper.body[1]
    assert isinstance(late_dispatch, ast.Return)
    late_call = next(
        node for node in ast.walk(late_dispatch) if isinstance(node, ast.Call)
    )
    assert isinstance(late_call.func, ast.Name)
    assert (
        late_call.func.id
        == "_optimize_sinet_late_residual_pre_add_mul_add_prelu_chains_pass"
    )
    assert {keyword.arg for keyword in late_call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }
    late_production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == late_wrapper_name
    ]
    assert len(late_production_calls) == 1
    late_layout_keyword = next(
        keyword
        for keyword in late_production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(late_layout_keyword.value, ast.Attribute)
    assert late_layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_deep_skip_owner_is_staged_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_deep_skip_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_tail(" in owner_source
    assert "def _resolve_stage(" in owner_source
    assert "def _resolve_branch(" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert "40" not in owner_source

    wrapper_name = (
        "_optimize_sinet_deep_skip_concat_resize_affine_tail_chains"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_deep_skip_concat_resize_affine_tail_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_preadd_fanout_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_preadd_fanout_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "_resolve_late_affine_tail(" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert '"40"' not in owner_source

    wrapper_name = (
        "_optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_dual_resize_owner_unifies_both_residual_modes() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_dual_resize_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_branch(" in owner_source
    assert "def _resolve_residual(" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert 'residual_mode="direct"' in owner_source
    assert 'residual_mode="sibling"' in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert '"40"' not in owner_source

    wrapper_aliases = {
        "_optimize_sinet_dual_resize_affine_transpose_chains": (
            "_optimize_sinet_dual_resize_affine_transpose_chains_pass"
        ),
        "_optimize_sinet_deep_skip_dual_resize_affine_transpose_chains": (
            "_optimize_sinet_deep_skip_dual_resize_affine_transpose_chains_pass"
        ),
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    for wrapper_name, alias_name in wrapper_aliases.items():
        wrapper = next(
            node
            for node in lowerer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        assert len(wrapper.body) == 2
        dispatch = wrapper.body[1]
        assert isinstance(dispatch, ast.Return)
        call = next(
            node for node in ast.walk(dispatch) if isinstance(node, ast.Call)
        )
        assert isinstance(call.func, ast.Name)
        assert call.func.id == alias_name
        assert {keyword.arg for keyword in call.keywords} == {
            "graph_index",
            "layout_state",
            "max_rewrites",
            "candidate",
        }
        production_calls = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == wrapper_name
        ]
        assert len(production_calls) == 1
        layout_keyword = next(
            keyword
            for keyword in production_calls[0].keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_shared_post_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_shared_post_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_input(" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _channel_last_concat(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert '"40"' not in owner_source

    wrapper_name = (
        "_optimize_sinet_shared_post_prelu_transpose_fanout_chains"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_shared_post_prelu_transpose_fanout_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_concat_resize_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_concat_resize_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_adapter(" in owner_source
    assert "def _resolve_affine_branch(" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert '"40"' not in owner_source

    wrapper_name = "_optimize_sinet_concat_resize_affine_transpose_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_concat_resize_affine_transpose_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_tail_concat_owner_reuses_indexed_branch_contracts() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_tail_concat_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "_resolve_adapter" in owner_source
    assert "_resolve_affine_branch" in owner_source
    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert '"40"' not in owner_source

    wrapper_name = (
        "_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_softmax_mask_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_softmax_mask_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "_NCHW_TO_NWHC" in owner_source
    assert "REDUCE_MAX" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert '"80"' not in owner_source

    wrapper_name = "_optimize_sinet_softmax_mask_residual_nhwc_tail_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_softmax_mask_residual_nhwc_tail_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_mix_attention_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_mix_attention_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "_NCKHW_TO_NHWCK" in owner_source
    assert "MIRROR_PAD" in owner_source
    assert "LOGISTIC" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source

    wrapper_name = "_optimize_sinet_mix_attention_double_logistic_nhwc_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_sinet_mix_attention_double_logistic_nhwc_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_sinet_sa_pa_mirrorpad_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "sinet_sa_pa_mirrorpad_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "_apply_constant_plans(" in owner_source
    assert "_apply_metadata_updates(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "_NCKHW_TO_NHWCK" in owner_source
    assert "MIRROR_PAD" in owner_source
    assert "nhwc_shape[3] != 1" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source

    wrapper_name = "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert (
        call.func.id
        == "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains_pass"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_transpose_binary_bridge_owner_is_bounded_and_transactional() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    owner_source = (pass_root / "binary_bridge_layout.py").read_text(
        encoding="utf-8"
    )
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_symmetric(" in owner_source
    assert "def _resolve_asymmetric(" in owner_source
    assert "def _apply_symmetric(" in owner_source
    assert "def _apply_asymmetric(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    assert "Pattern C" not in owner_source
    assert "enable_fanout_pattern_c" not in owner_source

    wrapper_name = "_optimize_transpose_binary_bridges"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 2
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "_optimize_transpose_binary_bridges_pass"
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert layout_keyword.value.attr == "layout_state"


def test_lowerer_gate_cluster_reuses_one_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_gate_layout_pass_cluster"
    )
    expected_order = [
        "run_mixed_attention_layout_cleanup",
        "run_elementwise_gate_layout_cleanup",
        "run_pad_layout_cleanup",
        "run_dual_postconv_gate_layout_cleanup",
        "run_ndhwc_gate_layout_cleanup",
        "run_cost_volume_scatter_layout_cleanup",
        "run_add_concat_suffix_layout_cleanup",
        "run_dual_mul_concat_layout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_gate_layout_pass_cluster"
    ]
    assert len(helper_invocations) == 2
    omitted_mixed_attention = [
        call
        for call in helper_invocations
        if any(
            keyword.arg == "include_mixed_attention"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
    ]
    assert len(omitted_mixed_attention) == 1


def test_lowerer_late_ndhwc_cost_volume_pair_reuses_one_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )

    assignment_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "late_ndhwc_cost_volume_state_scope"
    )
    assignment = lowerer.body[assignment_index]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "ModelIRPassStateScope"

    def _statement_call(statement: ast.stmt) -> ast.Call:
        assert isinstance(statement, ast.Expr)
        assert isinstance(statement.value, ast.Call)
        assert isinstance(statement.value.func, ast.Name)
        return statement.value

    mixed_call = _statement_call(lowerer.body[assignment_index - 2])
    raw_boundary_call = _statement_call(lowerer.body[assignment_index - 1])
    ndhwc_call = _statement_call(lowerer.body[assignment_index + 1])
    cost_volume_call = _statement_call(lowerer.body[assignment_index + 2])
    next_raw_boundary_call = _statement_call(lowerer.body[assignment_index + 3])

    assert mixed_call.func.id == "run_mixed_attention_layout_cleanup"
    assert all(keyword.arg != "state_scope" for keyword in mixed_call.keywords)
    assert (
        raw_boundary_call.func.id
        == "_optimize_transpose_dequant_hardsigmoid_quantize_bridges"
    )
    assert ndhwc_call.func.id == "run_ndhwc_gate_layout_cleanup"
    assert cost_volume_call.func.id == "run_cost_volume_scatter_layout_cleanup"
    assert next_raw_boundary_call.func.id == "_optimize_fold_conv_mul_add_affine_chains"

    for call in [ndhwc_call, cost_volume_call]:
        scope_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "late_ndhwc_cost_volume_state_scope"


def test_lowerer_late_concat_layout_cluster_reuses_one_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )

    assignment_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "late_concat_layout_state_scope"
    )
    assignment = lowerer.body[assignment_index]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "ModelIRPassStateScope"

    def _statement_call(statement: ast.stmt) -> ast.Call:
        assert isinstance(statement, ast.Expr)
        assert isinstance(statement.value, ast.Call)
        assert isinstance(statement.value.func, ast.Name)
        return statement.value

    previous_raw_boundary = _statement_call(lowerer.body[assignment_index - 1])
    assert previous_raw_boundary.func.id == "_optimize_fold_conv_mul_add_affine_chains"

    expected_order = [
        "run_axis3_const_concat_layout_cleanup",
        "run_dequant_concat_quantize_layout_cleanup",
        "run_layernorm_statistics_layout_cleanup",
        "run_layout_transpose_cleanup",
    ]
    runner_calls = [
        _statement_call(lowerer.body[assignment_index + offset])
        for offset in range(1, 5)
    ]
    assert [call.func.id for call in runner_calls] == expected_order
    for call in runner_calls:
        scope_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "late_concat_layout_state_scope"

    next_boundary = lowerer.body[assignment_index + 5]
    assert isinstance(next_boundary, ast.If)
    next_raw_boundary = _statement_call(next_boundary.body[0])
    assert (
        next_raw_boundary.func.id
        == "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"
    )


def test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )

    def _assert_helper(
        helper_name: str,
        expected_order: list[str],
    ) -> None:
        helper = next(
            node
            for node in lowerer.body
            if isinstance(node, ast.FunctionDef) and node.name == helper_name
        )
        calls = {
            node.func.id: node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in expected_order
        }
        assert [
            call.func.id
            for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
        ] == expected_order
        for name in expected_order:
            scope_keyword = next(
                keyword
                for keyword in calls[name].keywords
                if keyword.arg == "state_scope"
            )
            assert isinstance(scope_keyword.value, ast.Name)
            assert scope_keyword.value.id == "state_scope"

    channel_helper_name = "_run_channel_shuffle_gather_layout_pass_cluster"
    _assert_helper(
        channel_helper_name,
        [
            "run_two_way_channel_shuffle_cleanup",
            "run_nhwc_channel_shuffle_cleanup",
            "run_nchw_channel_shuffle_cleanup",
            "run_transpose_gather_axis_cleanup",
            "run_layout_transpose_cleanup",
            "run_transpose_unary_fanout_bridge_cleanup",
            "run_transpose_unary_binary_fanout_bridge_cleanup",
        ],
    )
    channel_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == channel_helper_name
    ]
    assert len(channel_invocations) == 3
    assert sum(
        any(
            keyword.arg == "include_post_gather_cleanup"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in channel_invocations
    ) == 1
    late_nchw_invocations = [
        call
        for call in channel_invocations
        if any(
            keyword.arg == "include_two_way_shuffle"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
    ]
    assert len(late_nchw_invocations) == 1
    assert any(
        keyword.arg == "include_nhwc_shuffle"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in late_nchw_invocations[0].keywords
    )

    unary_helper_name = "_run_transpose_unary_fanout_layout_pass_cluster"
    _assert_helper(
        unary_helper_name,
        [
            "run_layout_transpose_cleanup",
            "run_transpose_unary_passthrough_cleanup",
            "run_transpose_unary_fanout_bridge_cleanup",
            "run_transpose_unary_binary_fanout_bridge_cleanup",
        ],
    )
    unary_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == unary_helper_name
    ]
    assert len(unary_invocations) == 2
    post_qdq_invocations = [
        call
        for call in unary_invocations
        if any(
            keyword.arg == "include_layout_transpose"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    ]
    assert len(post_qdq_invocations) == 1
    assert any(
        keyword.arg == "include_unary_passthrough"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in post_qdq_invocations[0].keywords
    )


def test_lowerer_late_nchw_shuffle_gather_pair_stays_between_raw_rewrites() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_channel_shuffle_gather_layout_pass_cluster"
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
        and any(
            keyword.arg == "include_two_way_shuffle"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in statement.value.keywords
        )
    )
    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
    )
    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains"
    )


def test_lowerer_post_qdq_unary_fanout_cluster_stays_after_recovery_suffix() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_transpose_unary_fanout_layout_pass_cluster"
    layout_recovery = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
            and any(
                keyword.arg == "include_layout_transpose"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            for node in ast.walk(statement)
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(layout_recovery.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == helper_name
        and any(
            keyword.arg == "include_layout_transpose"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in statement.value.keywords
        )
    )
    previous_boundary = layout_recovery.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert (
        previous_boundary.value.func.id
        == "_run_layout_attention_quantized_recovery_suffix"
    )
    next_boundary = layout_recovery.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.Expr)
    assert isinstance(next_boundary.value, ast.Call)
    assert isinstance(next_boundary.value.func, ast.Name)
    assert (
        next_boundary.value.func.id
        == "_run_safe_binary_bridge_recovery_sequence"
    )


def test_lowerer_boundary_batchmatmul_unary_pair_reuses_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_boundary_batchmatmul_unary_layout_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_boundary_input_batchmatmul_cleanup",
        "run_input_unary_passthrough_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 1


def test_lowerer_channel_slice_pad_mul_pair_reuses_pass_state_scope() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper_name = "_run_channel_slice_pad_mul_layout_pass_cluster"
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    expected_order = [
        "run_channel_slice_merge_layout_cleanup",
        "run_pad_mul_layout_cleanup",
    ]
    calls = {
        node.func.id: node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_order
    }

    assert [
        call.func.id
        for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
    ] == expected_order
    for name in expected_order:
        scope_keyword = next(
            keyword
            for keyword in calls[name].keywords
            if keyword.arg == "state_scope"
        )
        assert isinstance(scope_keyword.value, ast.Name)
        assert scope_keyword.value.id == "state_scope"

    helper_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_invocations) == 2


def test_lowerer_singleton_reshape_clusters_reuse_pass_state_scopes() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )

    def _helper_calls(
        helper_name: str,
        expected_order: list[str],
    ) -> ast.FunctionDef:
        helper = next(
            node
            for node in lowerer.body
            if isinstance(node, ast.FunctionDef) and node.name == helper_name
        )
        calls = {
            node.func.id: node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in expected_order
        }
        assert [
            call.func.id
            for call in sorted(calls.values(), key=lambda candidate: candidate.lineno)
        ] == expected_order
        for name in expected_order:
            scope_keyword = next(
                keyword
                for keyword in calls[name].keywords
                if keyword.arg == "state_scope"
            )
            assert isinstance(scope_keyword.value, ast.Name)
            assert scope_keyword.value.id == "state_scope"
        return helper

    long_helper_name = "_run_singleton_reshape_layout_pass_cluster"
    _helper_calls(
        long_helper_name,
        [
            "run_layout_transpose_cleanup",
            "run_singleton_channel_transpose_cleanup",
            "run_duplicate_fanout_cleanup",
            "run_singleton_reshape_layout_cleanup",
            "run_singleton_maxpool_layout_cleanup",
            "run_flatten_concat_reshape_cleanup",
            "run_consecutive_reshape_cleanup",
            "run_squeeze_reshape_identity_cleanup",
            "run_singleton_spatial_reshape_cleanup",
            "run_multi_branch_gate_layout_cleanup",
        ],
    )
    long_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == long_helper_name
    ]
    assert len(long_invocations) == 2
    assert sum(
        any(
            keyword.arg == "include_layout_transpose"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        and any(
            keyword.arg == "include_multi_branch_gate"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in long_invocations
    ) == 1
    assert sum(
        any(
            keyword.arg == "include_duplicate_fanout"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        and any(
            keyword.arg == "include_spatial_concat_post_transpose"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
        for call in long_invocations
    ) == 1

    short_helper_name = "_run_singleton_consecutive_reshape_pass_cluster"
    short_helper = _helper_calls(
        short_helper_name,
        [
            "run_singleton_channel_transpose_cleanup",
            "run_duplicate_fanout_cleanup",
            "run_consecutive_reshape_cleanup",
        ],
    )
    assert [argument.arg for argument in short_helper.args.args] == [
        "target_model_ir",
        "target_layout_state",
    ]
    short_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == short_helper_name
    ]
    assert len(short_invocations) == 3
    assert sum(
        len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "model_ir"
        and isinstance(call.args[1], ast.Attribute)
        and call.args[1].attr == "layout_state"
        for call in short_invocations
    ) == 2
    assert sum(
        len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "fallback_ir"
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value is None
        for call in short_invocations
    ) == 1


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


def test_custom_op_artifact_metadata_has_single_scan_owner() -> None:
    builder_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py"
    ).read_text(encoding="utf-8")
    metadata_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "artifact_metadata.py"
    ).read_text(encoding="utf-8")

    assert "def collect_custom_op_artifact_metadata(" in metadata_source
    assert "def collect_custom_op_artifact_metadata(" not in builder_source
    assert "collect_custom_op_artifact_metadata(" in builder_source
    assert "custom_op_nodes_seen" not in builder_source


def test_tflite_evaluation_artifact_selection_has_single_owner() -> None:
    compatibility_source = (
        REPO_ROOT / "onnx2tf" / "onnx2tf.py"
    ).read_text(encoding="utf-8")
    metadata_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "artifact_metadata.py"
    ).read_text(encoding="utf-8")

    helper_name = "select_tflite_evaluation_artifact_paths"
    assert f"def {helper_name}(" in metadata_source
    assert f"def {helper_name}(" not in compatibility_source
    assert compatibility_source.count(f"{helper_name}(") == 1
    assert "direct_eval_paths = {}" not in compatibility_source
    assert "direct_eval_paths['" not in compatibility_source


def test_direct_report_and_quantization_finalization_has_single_owner() -> None:
    compatibility_source = (
        REPO_ROOT / "onnx2tf" / "onnx2tf.py"
    ).read_text(encoding="utf-8")
    compatibility_tree = ast.parse(compatibility_source)
    convert_function = next(
        node
        for node in compatibility_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "convert"
    )
    helper_name = (
        "_validate_and_log_flatbuffer_direct_reports_and_quantization"
    )
    helper = next(
        node
        for node in convert_function.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    helper_calls = [
        node
        for node in ast.walk(convert_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    assert len(helper_calls) == 1
    assert helper_calls[0].keywords == []

    helper_names = {
        node.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert {
        "report_op_coverage",
        "output_dynamic_range_quantized_tflite",
        "output_integer_quantized_tflite",
    } <= helper_names
    required_artifact_keys = {
        "op_coverage_report_path",
        "tensor_correspondence_report_path",
        "dynamic_range_quant_tflite_path",
        "integer_quant_tflite_path",
        "full_integer_quant_tflite_path",
        "integer_quant_with_int16_act_tflite_path",
        "full_integer_quant_with_int16_act_tflite_path",
    }
    helper_strings = {
        node.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert required_artifact_keys <= helper_strings
    assert "Dynamic Range Quantization tflite output complete! (" in helper_strings
    for error_message in (
        "flatbuffer_direct OP coverage report was requested but no report was generated.",
        "flatbuffer_direct dynamic-range quantization was requested but no output was generated.",
        "flatbuffer_direct integer quantization was requested but no output was generated.",
        "flatbuffer_direct full integer quantization was requested but no output was generated.",
        "flatbuffer_direct integer quantization with int16 activations was requested but no output was generated.",
        "flatbuffer_direct full integer quantization with int16 activations was requested but no output was generated.",
    ):
        assert error_message in helper_strings
        assert compatibility_source.count(error_message) == 1


def test_direct_backend_exits_before_legacy_tensorflow_pipeline() -> None:
    compatibility_source = (
        REPO_ROOT / "onnx2tf" / "onnx2tf.py"
    ).read_text(encoding="utf-8")
    compatibility_tree = ast.parse(compatibility_source)
    convert_function = next(
        node
        for node in compatibility_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "convert"
    )
    direct_fast_path = next(
        node
        for node in convert_function.body
        if isinstance(node, ast.If)
        and any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Name)
            and descendant.func.id == "_finalize_flatbuffer_direct_export"
            for descendant in ast.walk(node)
        )
    )
    fast_path_index = convert_function.body.index(direct_fast_path)
    boundary_assert = convert_function.body[fast_path_index + 1]

    assert isinstance(boundary_assert, ast.Assert)
    assert isinstance(boundary_assert.test, ast.Compare)
    assert isinstance(boundary_assert.test.left, ast.Name)
    assert boundary_assert.test.left.id == "tflite_backend"
    assert len(boundary_assert.test.ops) == 1
    assert isinstance(boundary_assert.test.ops[0], ast.Eq)
    assert len(boundary_assert.test.comparators) == 1
    assert isinstance(boundary_assert.test.comparators[0], ast.Constant)
    assert boundary_assert.test.comparators[0].value == "tf_converter"

    fast_path_returns = [
        node for node in ast.walk(direct_fast_path) if isinstance(node, ast.Return)
    ]
    assert len(fast_path_returns) == 1
    assert isinstance(fast_path_returns[0].value, ast.Constant)
    assert fast_path_returns[0].value.value is None

    direct_export_calls = [
        node
        for node in ast.walk(convert_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "export_tflite_model_flatbuffer_direct"
    ]
    assert len(direct_export_calls) == 1
    assert direct_export_calls[0].lineno < direct_fast_path.lineno

    legacy_tail = ast.Module(
        body=convert_function.body[fast_path_index + 2 :],
        type_ignores=[],
    )
    direct_backend_comparisons = [
        node
        for node in ast.walk(legacy_tail)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(descendant, ast.Name)
            and descendant.id == "tflite_backend"
            for descendant in ast.walk(node)
        )
        and any(
            isinstance(descendant, ast.Constant)
            and descendant.value == "flatbuffer_direct"
            for descendant in ast.walk(node)
        )
    ]
    assert direct_backend_comparisons == []
    assert "TF conversion path failed. " not in compatibility_source
    assert "flatbuffer_direct conversion failed." not in compatibility_source
    assert (
        "Skipping saved_model export and continuing with flatbuffer_direct flow."
        not in compatibility_source
    )


def test_direct_export_reads_options_only_from_normalized_request() -> None:
    builder_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py"
    ).read_text(encoding="utf-8")
    export_function = next(
        node
        for node in ast.parse(builder_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_tflite_model_flatbuffer_direct"
    )
    parent_by_id = {
        id(child): parent
        for parent in ast.walk(export_function)
        for child in ast.iter_child_nodes(parent)
    }
    kwargs_reads = [
        node
        for node in ast.walk(export_function)
        if isinstance(node, ast.Name)
        and node.id == "kwargs"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(kwargs_reads) == 1
    boundary_call = parent_by_id[id(kwargs_reads[0])]
    assert isinstance(boundary_call, ast.Call)
    assert isinstance(boundary_call.func, ast.Attribute)
    assert isinstance(boundary_call.func.value, ast.Name)
    assert boundary_call.func.value.id == "ConversionRequest"
    assert boundary_call.func.attr == "from_kwargs"

    raw_kwargs_gets = [
        node
        for node in ast.walk(export_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kwargs"
        and node.func.attr == "get"
    ]
    assert raw_kwargs_gets == []
    request_gets = [
        node
        for node in ast.walk(export_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "request"
        and node.func.attr == "get"
    ]
    assert request_gets


def test_op_coverage_writer_is_called_only_when_requested() -> None:
    builder_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py"
    ).read_text(encoding="utf-8")
    builder_tree = ast.parse(builder_source)
    export_function = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_tflite_model_flatbuffer_direct"
    )
    parent_by_id = {
        id(child): parent
        for parent in ast.walk(export_function)
        for child in ast.iter_child_nodes(parent)
    }
    report_calls = [
        node
        for node in ast.walk(export_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_coverage_report"
    ]
    assert len(report_calls) == 3
    for report_call in report_calls:
        ancestor = parent_by_id.get(id(report_call))
        guarded = False
        while ancestor is not None and ancestor is not export_function:
            if isinstance(ancestor, ast.If) and isinstance(ancestor.test, ast.Name):
                if ancestor.test.id == "report_op_coverage":
                    guarded = True
                    break
            ancestor = parent_by_id.get(id(ancestor))
        assert guarded


def test_saved_model_progress_advances_only_when_artifact_is_requested() -> None:
    builder_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py"
    ).read_text(encoding="utf-8")
    export_function = next(
        node
        for node in ast.parse(builder_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_tflite_model_flatbuffer_direct"
    )
    saved_model_guard = next(
        node
        for node in ast.walk(export_function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "output_saved_model_from_model_ir"
    )
    last_statement = saved_model_guard.body[-1]
    assert isinstance(last_statement, ast.Expr)
    assert isinstance(last_statement.value, ast.Call)
    assert isinstance(last_statement.value.func, ast.Name)
    assert last_statement.value.func.id == "_advance_export_progress"


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
    pass_source = pass_path.read_text(encoding="utf-8")
    assert "model_ir.operators =" not in pass_source
    assert "op.inputs =" not in pass_source
    assert "op.outputs =" not in pass_source
    assert "ModelIRGraphIndex" in pass_source

    assert "_prune_unused_tensors" in common_functions
    assert "_is_fully_known_positive_shape" in common_functions
    assert "_broadcast_shape_signatures" in common_functions
    for path in (lowering_path, precision_path, constant_fold_path):
        functions = _functions(path)
        assert "_prune_unused_tensors" not in functions
        assert "_is_fully_known_positive_shape" not in functions
        assert "_broadcast_shape_signatures" not in functions


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


def test_mul_square_constant_fold_has_generic_indexed_owner() -> None:
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

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowering_functions = _functions(lowering_path)
    pass_functions = _functions(pass_path)
    owner_name = "_optimize_mul_square_anchor_constant_chains"
    wrapper_name = "_optimize_yolo_decode_mul_square_anchor_chains"
    assert owner_name in pass_functions
    assert "yolo" not in owner_name
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[wrapper_name])
        if isinstance(node, ast.Name)
    }
    assert f"{wrapper_name}_pass" in wrapper_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(pass_functions[owner_name])
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "producer" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "remove_operators" in owner_calls


def test_gather_reshape_cleanup_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "gather_reshape_cleanup.py"
    )
    function_name = (
        "_optimize_gather_axis0_singleton_to_reshape_input_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_prune_unused_tensors" in owner_calls


def test_terminal_softmax_layout_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_softmax_layout.py"
    )
    function_name = (
        "_optimize_terminal_softmax_transpose_after_nhwc_propagation"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_prune_unused_tensors" in owner_calls
    assert (
        "__softmax_nhwc_propagated__"
        not in lowering_path.read_text(encoding="utf-8")
    )
    assert (
        pass_path.read_text(encoding="utf-8").count(
            "__softmax_nhwc_propagated__"
        )
        == 1
    )


def test_terminal_argmax_layout_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_argmax_layout.py"
    )
    function_name = "_optimize_transpose_pre_argmax_nhwc_terminal_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_prune_unused_tensors" in owner_calls


def test_quantized_pool_cleanup_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_pool.py"
    )
    function_name = "_optimize_dequant_maxpool_quantize_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_prune_unused_tensors" in owner_calls


def test_quantized_logistic_cleanup_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_logistic.py"
    )
    function_name = "_optimize_dequant_logistic_quantize_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_prune_unused_tensors" in owner_calls


def test_quantized_softmax_cleanup_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_softmax.py"
    )
    function_name = "_optimize_dequant_softmax_quantize_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_prune_unused_tensors" in owner_calls


def test_quantized_hardsigmoid_fold_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_hardsigmoid.py"
    )
    function_name = "_optimize_dequant_hardsigmoid_quantize_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_prune_unused_tensors" in owner_calls


def test_quantized_transpose_conv_cleanup_has_indexed_semantic_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_transpose_conv.py"
    )
    function_name = "_optimize_dequant_transposeconv_quantize_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_function = _functions(pass_path)[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(owner_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_prune_unused_tensors" in owner_calls


def test_decomposed_instance_normalization_repair_has_indexed_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "instance_normalization_layout.py"
    )
    function_name = "_repair_decomposed_instance_normalization_layouts"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (owner_function, owner_functions["_candidate_plans"])
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_apply" in owner_calls
    assert "sync_from_model_ir" in owner_calls


def test_concat_global_pool_axis_repair_has_indexed_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "concat_global_pool_layout.py"
    )
    function_name = "_repair_nchw_concat_global_pool_conv_axes"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_function = _functions(lowering_path)[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    lowerer_names = {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in lowerer_names

    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (owner_function, owner_functions["_candidate_plan"])
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls


def test_concat_transpose_conv_axis_repair_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "concat_transpose_conv_layout.py"
    )
    function_name = "_repair_nchw_concat_transpose_conv_axes"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_function = _functions(lowering_path)[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_candidate_plan"],
            owner_functions["_chain_fanout_is_compatible"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices_for_types" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls


def test_mixed_singleton_concat_repair_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "mixed_singleton_concat_layout.py"
    )
    function_name = "_repair_mixed_singleton_nchw_inputs_for_nhwc_concat"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_function = _functions(lowering_path)[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_candidate_plan"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "insert_operator" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls


def test_window_partition_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "window_partition_layout.py"
    )
    function_name = (
        "_optimize_window_partition_reshape_transpose_to_space_to_depth_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_function = _functions(lowering_path)[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_candidate_plan"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "replace_operator_type" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls


def test_window_reverse_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "window_partition_layout.py"
    )
    function_name = (
        "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_reverse_candidate_plan"],
            owner_functions["_apply_reverse_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "replace_operator_type" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "_apply_reverse_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 2
    for call in production_calls:
        layout_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_conv1d_unary_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv1d_unary_layout.py"
    )
    function_name = (
        "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_candidate"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "_resolve_unary_prefix_candidate" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_rank4_conv1d_unary_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv1d_unary_layout.py"
    )
    function_name = (
        "_optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_rank4_candidate"],
            owner_functions["_apply_rank4_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "insert_operator" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_replace_tensor_inputs" in owner_calls
    assert "_apply_rank4_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_conv1d_unary_fanout_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv1d_unary_layout.py"
    )
    function_name = (
        "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_fanout_candidate"],
            owner_functions["_apply_fanout_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "_resolve_unary_prefix_candidate" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "insert_operator" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "_apply_fanout_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_conv1d_instance_norm_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv1d_instance_norm_layout.py"
    )
    function_name = (
        "_optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_candidate"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_replace_tensor_inputs" in owner_calls
    assert "_plan_constant_update" in owner_calls
    assert "_apply_constant_update" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_conv1d_tencoder_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv1d_tencoder_layout.py"
    )
    function_name = "_optimize_tencoder_add_expand_transpose_conv_nhwc_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_simple_lhs"],
            owner_functions["_resolve_legacy_lhs"],
            owner_functions["_resolve_gate"],
            owner_functions["_resolve_candidate"],
            owner_functions["_plan_scale_constant_update"],
            owner_functions["_plan_int_update"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "operator_index" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "insert_operator" in owner_calls
    assert "remove_operators" in owner_calls
    assert "_resolve_flattened_instance_norm_prefix" in owner_calls
    assert "_plan_constant_update" in owner_calls
    assert "_apply_constant_update" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_replace_tensor_inputs" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_conv1d_batchmatmul_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv1d_batchmatmul_layout.py"
    )
    function_name = "_optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains"

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_candidate"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "operator_index" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_decoder_deconv_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "decoder_deconv_layout.py"
    )
    function_name = (
        "_optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id for node in ast.walk(lowerer_function) if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_candidate"],
            owner_functions["_plan_bias_update"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "operator_index" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_plan_constant_update" in owner_calls
    assert "_apply_constant_update" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_terminal_squeeze_mean_layout_rewrite_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_squeeze_mean_layout.py"
    )
    function_name = (
        "_optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_functions = _functions(lowering_path)
    lowerer_function = lowerer_functions[function_name]
    owner_functions = _functions(pass_path)
    owner_function = owner_functions[function_name]
    assert f"{function_name}_pass" in {
        node.id
        for node in ast.walk(lowerer_function)
        if isinstance(node, ast.Name)
    }
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner_function,
            owner_functions["_resolve_candidate"],
            owner_functions["_apply_plan"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "operator_index" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "remove_operator" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_apply_constant_update" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_instance_norm_direct_prepost_layout_has_indexed_owner() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "instance_norm_prepost_layout.py"
    )
    common_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "decomposed_instance_norm.py"
    )
    compatibility_name = "_optimize_transpose_instancenorm_prepost_nhwc_chains"
    owner_name = (
        "_optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains"
    )
    side_owner_name = (
        "_optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains"
    )
    unary_reshape_owner_name = (
        "_optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains"
    )
    residual_reshape_owner_name = (
        "_optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    lowerer_functions = _functions(lowering_path)
    compatibility = lowerer_functions[compatibility_name]
    assert compatibility.end_lineno is not None
    assert compatibility.end_lineno - compatibility.lineno + 1 <= 70
    compatibility_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(compatibility)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert {
        "_build_tensor_consumer_map",
        "_build_tensor_producer_map",
        "_prune_unused_tensors",
        "_replace_operator_input_at",
        "_set_operator_inputs",
        "_set_operator_outputs",
        "insert_operator",
        "remove_operators",
    }.isdisjoint(compatibility_calls)
    assert f"{owner_name}_pass" in {
        node.id
        for node in ast.walk(compatibility)
        if isinstance(node, ast.Name)
    }
    assert f"{side_owner_name}_pass" in {
        node.id
        for node in ast.walk(compatibility)
        if isinstance(node, ast.Name)
    }
    assert f"{unary_reshape_owner_name}_pass" in {
        node.id
        for node in ast.walk(compatibility)
        if isinstance(node, ast.Name)
    }
    assert f"{residual_reshape_owner_name}_pass" in {
        node.id
        for node in ast.walk(compatibility)
        if isinstance(node, ast.Name)
    }
    owner_functions = _functions(pass_path)
    common_functions = _functions(common_path)
    owner = owner_functions[owner_name]
    owner_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for owner_node in (
            owner,
            owner_functions[side_owner_name],
            owner_functions[unary_reshape_owner_name],
            owner_functions[residual_reshape_owner_name],
            owner_functions["_run_indexed_instance_norm_prepost_tail"],
            owner_functions["_resolve_candidate"],
            owner_functions["_resolve_residual_tail"],
            owner_functions["_resolve_residual_source"],
            owner_functions["_apply_plan"],
            common_functions["match_decomposed_instance_norm_core"],
            common_functions["plan_constant_update"],
            common_functions["apply_constant_update"],
        )
        for node in ast.walk(owner_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in owner_calls
    assert "_build_tensor_producer_map" not in owner_calls
    assert "ModelIRGraphIndex" in owner_calls
    assert "operator_indices" in owner_calls
    assert "operator_index" in owner_calls
    assert "consumer_indices" in owner_calls
    assert "remove_operators" in owner_calls
    assert "insert_operator" in owner_calls
    assert "_set_operator_inputs" in owner_calls
    assert "_set_operator_outputs" in owner_calls
    assert "_replace_operator_input_at" in owner_calls
    assert "_apply_plan" in owner_calls
    assert "sync_from_model_ir" in owner_calls

    production_calls = [
        node
        for node in ast.walk(lowerer_functions["lower_onnx_to_ir"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == compatibility_name
    ]
    assert len(production_calls) == 2
    for call in production_calls:
        layout_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_precision_rewrites_use_differential_graph_index() -> None:
    precision_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "precision.py"
    )
    source = precision_path.read_text(encoding="utf-8")
    assert "_build_tensor_consumer_map" not in source
    assert "model_ir.operators =" not in source
    assert "op.op_type =" not in source
    assert "op.inputs =" not in source


def test_high_rank_binary_rewrite_uses_differential_graph_index() -> None:
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "high_rank_binary.py"
    )
    source = pass_path.read_text(encoding="utf-8")
    assert "model_ir.operators =" not in source
    assert "ModelIRGraphIndex" in source


def test_pytorch_compat_and_control_flow_have_focused_owners() -> None:
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pytorch_compat.py"
    )
    source = pass_path.read_text(encoding="utf-8")
    assert "model_ir.operators =" not in source
    assert "op.outputs =" not in source
    assert "ModelIRGraphIndex" in source
    exporter_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    )
    control_flow_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pytorch_control_flow.py"
    )
    recurrent_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pytorch_recurrent.py"
    )
    normalization_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pytorch_normalization.py"
    )
    exporter_source = exporter_path.read_text(encoding="utf-8")
    exporter_tree = ast.parse(exporter_source)
    control_flow_source = control_flow_path.read_text(encoding="utf-8")
    control_flow_tree = ast.parse(control_flow_source)
    recurrent_source = recurrent_path.read_text(encoding="utf-8")
    recurrent_tree = ast.parse(recurrent_source)
    normalization_source = normalization_path.read_text(encoding="utf-8")
    normalization_tree = ast.parse(normalization_source)
    assert "def _remove_redundant_layout_transposes(" not in exporter_source
    assert "_remove_redundant_layout_transposes," not in exporter_source
    assert "_remove_redundant_layout_transposes," in normalization_source
    assert "def _rewrite_atan2_ones_like_to_atan(" not in exporter_source
    assert "_rewrite_atan2_ones_like_to_atan," not in exporter_source
    assert "_rewrite_atan2_ones_like_to_atan," in normalization_source
    assert "def _rewrite_atan2_ones_like_to_atan(" in source
    assert "replace_operator_type(" in source
    assert "replace_operator_inputs(" in source
    assert "def _reject_residual_layout_transposes(" not in exporter_source
    assert "_reject_residual_layout_transposes," not in exporter_source
    assert "_reject_residual_layout_transposes," in normalization_source
    assert "def _is_reshape_only_residual_layout_bridge_transpose(" not in exporter_source
    assert "_is_reshape_only_residual_layout_bridge_transpose," not in exporter_source
    emitter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_emitters.py"
    ).read_text(encoding="utf-8")
    assert "_is_reshape_only_residual_layout_bridge_transpose," in emitter_source
    assert "def _reject_residual_layout_transposes(" in source
    assert 'operator_indices("TRANSPOSE")' in source
    operator_stream_assignments = [
        node
        for tree in (exporter_tree, control_flow_tree, normalization_tree)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (
            list(node.targets)
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Attribute) and target.attr == "operators"
    ]
    assert operator_stream_assignments == []
    assert "def _rewrite_static_while_ops_for_native_export(" not in exporter_source
    assert "def _rewrite_counter_bounded_while_ops_for_native_export(" not in exporter_source
    assert "_rewrite_static_while_ops_for_native_export," not in exporter_source
    assert "_rewrite_counter_bounded_while_ops_for_native_export," not in exporter_source
    assert "_rewrite_static_while_ops_for_native_export," in normalization_source
    assert "_rewrite_counter_bounded_while_ops_for_native_export," in normalization_source
    assert "def _rewrite_recurrent_ops_for_native_export(" not in exporter_source
    assert "_rewrite_recurrent_ops_for_native_export," not in exporter_source
    assert "_rewrite_recurrent_ops_for_native_export," in normalization_source
    assert "def _clone_model_ir_without_root_operators(" in control_flow_source
    assert "for op_index, source_op in enumerate(model_ir.operators):" in control_flow_source
    assert "ModelIRGraphIndex" in control_flow_source
    assert "for candidate in body_subgraph.operators:" not in control_flow_source

    exporter_functions = {
        node.name: node
        for node in exporter_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    control_flow_functions = {
        node.name: node
        for node in control_flow_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    focused_control_flow_functions = {
        "_clone_model_ir_without_root_operators",
        "_get_model_ir_subgraph_by_1based_index",
        "_constant_scalar_value",
        "_reshape_alias_source_name",
        "_is_canonical_imported_while_cond_subgraph",
        "_match_static_unrollable_while_op",
        "_match_counter_bounded_unrollable_while_op",
        "_ensure_tensor_shape_literal",
        "_rewrite_static_while_ops_for_native_export",
        "_rewrite_counter_bounded_while_ops_for_native_export",
    }
    assert focused_control_flow_functions <= set(control_flow_functions)
    assert focused_control_flow_functions.isdisjoint(exporter_functions)
    copy_on_write_functions = {
        "_rewrite_static_while_ops_for_native_export",
        "_rewrite_counter_bounded_while_ops_for_native_export",
    }
    matcher_by_rewriter = {
        "_rewrite_static_while_ops_for_native_export": (
            "_match_static_unrollable_while_op"
        ),
        "_rewrite_counter_bounded_while_ops_for_native_export": (
            "_match_counter_bounded_unrollable_while_op"
        ),
    }
    for function_name in copy_on_write_functions:
        function_source = ast.get_source_segment(
            control_flow_source,
            control_flow_functions[function_name],
        )
        assert function_source is not None
        assert "return copy.deepcopy(model_ir)" not in function_source
        assert "return model_ir" in function_source
        assert function_source.count(matcher_by_rewriter[function_name]) == 1
        assert "rewrite_plans.get(int(op_index), None)" in function_source
    recurrent_functions = {
        node.name: node
        for node in recurrent_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    focused_recurrent_functions = {
        "_repair_orphan_recurrent_step_tensors",
        "_sequence_lstm_input_name",
        "_tensor_has_constant_data",
        "_sequence_lstm_bias_inputs_supported",
        "_sequence_lstm_index_spec",
        "_can_direct_codegen_sequence_lstm_op",
        "_can_direct_codegen_sequence_rnn_op",
        "_rewrite_recurrent_ops_for_native_export",
    }
    assert focused_recurrent_functions <= set(recurrent_functions)
    assert focused_recurrent_functions.isdisjoint(exporter_functions)
    recurrent_rewrite_source = ast.get_source_segment(
        recurrent_source,
        recurrent_functions["_rewrite_recurrent_ops_for_native_export"],
    )
    assert recurrent_rewrite_source is not None
    assert "return copy.deepcopy(model_ir)" not in recurrent_rewrite_source
    assert "return model_ir" in recurrent_rewrite_source
    assert recurrent_rewrite_source.count("for op in model_ir.operators") == 1
    assert "if not any(" not in recurrent_rewrite_source
    assert "if all(" not in recurrent_rewrite_source
    orphan_repair_source = ast.get_source_segment(
        recurrent_source,
        recurrent_functions["_repair_orphan_recurrent_step_tensors"],
    )
    assert orphan_repair_source is not None
    assert "repair_orphan_recurrent_step_tensors(" in orphan_repair_source
    assert "graph_index=graph_index" in orphan_repair_source
    assert "graph_index.replace_operator_inputs(" not in orphan_repair_source
    assert "for op in model_ir.operators" not in orphan_repair_source


def test_pytorch_softmax_layout_validation_reuses_one_graph_index() -> None:
    exporter_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pytorch_layout_validation.py"
    )
    layout_utils_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_layout_utils.py"
    )
    normalization_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pytorch_normalization.py"
    )
    exporter_source = exporter_path.read_text(encoding="utf-8")
    pass_source = pass_path.read_text(encoding="utf-8")
    layout_utils_source = layout_utils_path.read_text(encoding="utf-8")
    normalization_source = normalization_path.read_text(encoding="utf-8")
    normalization_tree = ast.parse(normalization_source)
    pass_tree = ast.parse(pass_source)
    assert "def _is_attention_like_softmax_op(" not in exporter_source
    assert (
        "def _is_transpose_sandwiched_last_axis_softmax_op(" not in exporter_source
    )
    assert "_is_attention_like_softmax_op," not in exporter_source
    assert "_is_transpose_sandwiched_last_axis_softmax_op," not in exporter_source
    assert "def _is_attention_like_softmax_op(" in pass_source
    assert "def _is_transpose_sandwiched_last_axis_softmax_op(" in pass_source
    assert "def validate_channel_first_exportability(" not in exporter_source
    assert "validate_channel_first_exportability," not in exporter_source
    assert "def validate_channel_first_exportability(" in pass_source
    assert "def _apply_feature_last_sequence_layouts(" not in exporter_source
    assert "_apply_feature_last_sequence_layouts," not in exporter_source
    assert "def _ensure_public_boundary_layout_bridges(" not in exporter_source
    assert "_ensure_public_boundary_layout_bridges," not in exporter_source
    assert "def _ensure_public_boundary_layout_bridges(" in pass_source
    assert "def _propagate_pytorch_friendly_layouts(" not in exporter_source
    assert "_propagate_pytorch_friendly_layouts," not in exporter_source
    assert "def _rewrite_filter_tensors_for_pytorch(" not in exporter_source
    assert "_rewrite_filter_tensors_for_pytorch," not in exporter_source
    assert "def _rewrite_layout_sensitive_ops(" not in exporter_source
    assert "_rewrite_layout_sensitive_ops," not in exporter_source
    assert (
        "def _synchronize_reshape_targets_with_output_tensors("
        not in exporter_source
    )
    assert (
        "_synchronize_reshape_targets_with_output_tensors,"
        not in exporter_source
    )
    assert "def _align_public_boundary_shapes_to_onnx_contract(" not in exporter_source
    assert "_align_public_boundary_shapes_to_onnx_contract," not in exporter_source
    assert "_align_public_boundary_shapes_to_onnx_contract," in normalization_source
    assert "def _align_public_boundary_shapes_to_onnx_contract(" in pass_source
    assert "def _has_recurrent_sequence_context(" not in exporter_source
    assert "def _has_recurrent_sequence_context(" in pass_source
    for function_name in {
        "_preferred_reshape_target_values",
        "_tensor_name_suggests_channel_last_layout_for_codegen",
    }:
        assert f"def {function_name}(" not in exporter_source
        assert f"{function_name}," in exporter_source
        assert f"def {function_name}(" in layout_utils_source
    assert "def _preferred_reshape_target_values_for_op(" not in exporter_source
    assert "_preferred_reshape_target_values_for_op," not in exporter_source
    assert "def _preferred_reshape_target_values_for_op(" in layout_utils_source
    assert "_preferred_reshape_target_values_for_op," in pass_source
    focused_layout_owner_functions = {
        "_collect_feature_last_sequence_tensor_names",
        "_is_pytorch_preserved_channel_last_rank4_or_rank5_model_island",
        "_shrink_preserved_channel_last_regions_for_pytorch",
        "_restore_non_preserved_channel_first_layouts",
    }
    for function_name in focused_layout_owner_functions:
        assert f"def {function_name}(" not in exporter_source
        if function_name == "_collect_feature_last_sequence_tensor_names":
            assert f"{function_name}," in exporter_source
        else:
            assert f"{function_name}," not in exporter_source
    assert "def _is_rank4_channel_last_dynamic_tensor(" not in exporter_source
    assert "ModelIRGraphIndex" in pass_source
    assert "for candidate in model_ir.operators" not in pass_source
    assert "def _propagate_feature_last_tensor_names(" in pass_source
    assert "def _propagate_channel_last_layouts(" in pass_source
    assert "while changed:" not in pass_source

    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    normalization_functions = {
        node.name: node
        for node in normalization_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_collect_model_op_types",
        "_is_layout_agnostic_native_model_ir",
        "_rewrite_native_pytorch_compatibility_ops",
        "_normalize_model_ir_for_pytorch_channel_first_with_index",
        "normalize_model_ir_for_pytorch_channel_first",
        "prepare_model_ir_for_native_pytorch",
    } <= set(normalization_functions)
    assert focused_layout_owner_functions | {
        "_is_pytorch_channel_first_safe_rank4_island_op",
        "_is_rank4_channel_last_dynamic_tensor",
    } <= set(pass_functions)
    validator_source = ast.get_source_segment(
        pass_source,
        pass_functions["validate_channel_first_exportability"],
    )
    assert validator_source is not None
    assert "graph_index = graph_index or ModelIRGraphIndex(model_ir)" in validator_source
    assert "operator_indices_for_types(" in validator_source
    assert "graph_index=graph_index" in validator_source
    assert "for op in model_ir.operators" not in validator_source
    normalizer_wrapper_source = ast.get_source_segment(
        normalization_source,
        normalization_functions["normalize_model_ir_for_pytorch_channel_first"],
    )
    normalizer_source = ast.get_source_segment(
        normalization_source,
        normalization_functions[
            "_normalize_model_ir_for_pytorch_channel_first_with_index"
        ],
    )
    assert normalizer_wrapper_source is not None
    assert normalizer_source is not None
    assert "def normalize_model_ir_for_pytorch_channel_first(" not in exporter_source
    assert "normalize_model_ir_for_pytorch_channel_first," in exporter_source
    assert (
        "_normalize_model_ir_for_pytorch_channel_first_with_index("
        in normalizer_wrapper_source
    )
    assert normalizer_source.count("ModelIRGraphIndex(normalized)") == 1
    assert "_build_model_ir_producer_consumer_index(normalized)" not in normalizer_source
    assert normalizer_source.count("graph_index=layout_graph_index") >= 7
    assert "consumers=layout_graph_index.consumers" in normalizer_source
    prepare_source = ast.get_source_segment(
        normalization_source,
        normalization_functions["prepare_model_ir_for_native_pytorch"],
    )
    assert prepare_source is not None
    assert "def prepare_model_ir_for_native_pytorch(" not in exporter_source
    assert "prepare_model_ir_for_native_pytorch," in exporter_source
    assert "_rewrite_native_pytorch_compatibility_ops(model_ir)" in prepare_source
    compatibility_source = ast.get_source_segment(
        normalization_source,
        normalization_functions["_rewrite_native_pytorch_compatibility_ops"],
    )
    assert compatibility_source is not None
    assert "_rewrite_static_while_ops_for_native_export(" in compatibility_source
    assert (
        "_rewrite_counter_bounded_while_ops_for_native_export("
        in compatibility_source
    )
    assert "_rewrite_recurrent_ops_for_native_export(" in compatibility_source
    assert "root_op_types" in compatibility_source
    assert (
        "_normalize_model_ir_for_pytorch_channel_first_with_index("
        in prepare_source
    )
    assert "boundary_graph_index = normalization_result.graph_index" in prepare_source
    assert prepare_source.count("ModelIRGraphIndex(prepared)") == 1
    assert "graph_index=boundary_graph_index" in prepare_source
    collector_source = ast.get_source_segment(
        pass_source,
        pass_functions["_collect_feature_last_sequence_tensor_names"],
    )
    assert collector_source is not None
    assert "graph_index = graph_index or ModelIRGraphIndex(model_ir)" in collector_source
    assert "_propagate_feature_last_tensor_names(" in collector_source
    assert "while changed:" not in collector_source
    apply_layout_source = ast.get_source_segment(
        pass_source,
        pass_functions["_apply_feature_last_sequence_layouts"],
    )
    assert apply_layout_source is not None
    assert "_propagate_channel_last_layouts(" in apply_layout_source
    assert "while changed:" not in apply_layout_source
    boundary_bridge_source = ast.get_source_segment(
        pass_source,
        pass_functions["_ensure_public_boundary_layout_bridges"],
    )
    assert boundary_bridge_source is not None
    assert "for op in model_ir.operators" not in boundary_bridge_source
    assert "replace_operator_inputs(" in boundary_bridge_source
    assert "replace_operator_outputs(" in boundary_bridge_source
    assert "insert_operator(" in boundary_bridge_source
    assert "append_operator(" in boundary_bridge_source
    friendly_layout_source = ast.get_source_segment(
        pass_source,
        pass_functions["_propagate_pytorch_friendly_layouts"],
    )
    assert friendly_layout_source is not None
    assert "operator_indices_for_types(" in friendly_layout_source
    assert "consumer_indices(" in friendly_layout_source
    assert "while changed:" not in friendly_layout_source
    filter_rewrite_source = ast.get_source_segment(
        pass_source,
        pass_functions["_rewrite_filter_tensors_for_pytorch"],
    )
    assert filter_rewrite_source is not None
    assert "operator_indices_for_types(" in filter_rewrite_source
    assert "for op in model_ir.operators" not in filter_rewrite_source
    sensitive_rewrite_source = ast.get_source_segment(
        pass_source,
        pass_functions["_rewrite_layout_sensitive_ops"],
    )
    assert sensitive_rewrite_source is not None
    assert "operator_indices_for_types(" in sensitive_rewrite_source
    assert "for op in model_ir.operators" not in sensitive_rewrite_source
    reshape_sync_source = ast.get_source_segment(
        pass_source,
        pass_functions["_synchronize_reshape_targets_with_output_tensors"],
    )
    assert reshape_sync_source is not None
    assert 'operator_indices("RESHAPE")' in reshape_sync_source
    assert "for op in model_ir.operators" not in reshape_sync_source
    recurrent_context_source = ast.get_source_segment(
        pass_source,
        pass_functions["_has_recurrent_sequence_context"],
    )
    assert recurrent_context_source is not None
    assert "operator_indices_for_types(" in recurrent_context_source


def test_dynamic_rank1_reshape_rewrite_has_indexed_pass_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "dynamic_reshape.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    pass_source = pass_path.read_text(encoding="utf-8")
    assert (
        "rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs as "
        "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs_pass"
    ) in lowering_source
    assert (
        "restore_placeholder_matmul_flattened_inputs as "
        "_restore_placeholder_matmul_flattened_inputs_pass"
    ) in lowering_source
    assert "return _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs_pass(" in (
        lowering_source
    )
    assert "model_ir.operators =" not in pass_source
    assert "op.inputs[1] =" not in pass_source
    assert "matmul_op.inputs[0] =" not in pass_source
    assert "ModelIRGraphIndex" in pass_source


def test_dead_operator_pruning_uses_batch_graph_index_compaction() -> None:
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
    graph_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "core" / "graph.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    pass_source = pass_path.read_text(encoding="utf-8")
    pass_tree = ast.parse(pass_source)
    prune_node = next(
        node
        for node in pass_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prune_dead_operators"
    )
    prune_source = ast.get_source_segment(pass_source, prune_node)
    assert prune_source is not None
    assert "prune_dead_operators as _prune_dead_operators_pass" in lowering_source
    assert "return _prune_dead_operators_pass(" in lowering_source
    assert "remove_operators(remove_indices)" in prune_source
    assert "model_ir.operators =" not in prune_source
    assert "def remove_operators(" in graph_path.read_text(encoding="utf-8")


def test_unsupported_split_fallback_has_indexed_pass_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "split_fallback.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    pass_source = pass_path.read_text(encoding="utf-8")
    assert (
        "replace_unsupported_split_with_slice as "
        "_replace_unsupported_split_with_slice_pass"
    ) in lowering_source
    assert "return _replace_unsupported_split_with_slice_pass(" in lowering_source
    assert "model_ir.operators =" not in pass_source
    assert "ModelIRGraphIndex" in pass_source
    assert "graph_index.remove_operator(" in pass_source
    assert "graph_index.insert_operator(" in pass_source


def test_expand_squeeze_pre_ops_use_differential_graph_index_insertion() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_replace_expand_dims_and_squeeze_with_reshape"
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_replace_expand_dims_and_squeeze_with_reshape_pass"
    ]
    assert len(dispatches) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "expand_squeeze_reshape.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "replace_expand_dims_and_squeeze_with_reshape"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    assert "model_ir.operators =" not in owner_function_source
    assert "ModelIRGraphIndex(model_ir)" in owner_function_source
    assert "graph_index.insert_operator(" in owner_function_source
    assert "_prune_unused_tensors(" in owner_function_source
    assert "layout_state.sync_from_model_ir(model_ir)" in owner_function_source
    assert "lower_from_onnx2tf" not in owner_source


def test_rank4_binary_layout_adapter_has_one_module_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper_name = "_repair_rank4_binary_layout_mismatch_with_transpose_adapter"
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_repair_rank4_binary_layout_mismatch_with_transpose_adapter_pass"
    ]
    assert len(dispatches) == 1

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 4

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "binary_layout_adapter.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "repair_rank4_binary_layout_mismatch_with_transpose_adapter"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    assert "model_ir.operators.insert(" in owner_function_source
    assert "_replace_operator_input_at(" in owner_function_source
    assert "_prune_unused_tensors(model_ir)" in owner_function_source
    assert "lower_from_onnx2tf" not in owner_source


def test_rank4_binary_singleton_adapter_has_one_module_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper_name = "_repair_rank4_binary_singleton_broadcast_layout_mismatch"
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_repair_rank4_binary_singleton_broadcast_layout_mismatch_pass"
    ]
    assert len(dispatches) == 1

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 4

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "binary_layout_adapter.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "repair_rank4_binary_singleton_broadcast_layout_mismatch"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    assert "np.broadcast_shapes" in owner_function_source
    assert 'op_type="TRANSPOSE"' in owner_function_source
    assert 'op_type="RESHAPE"' in owner_function_source
    assert "_replace_operator_input_at(" in owner_function_source
    assert "if repaired > 0:" in owner_function_source
    assert "_prune_unused_tensors(model_ir)" in owner_function_source
    assert "lower_from_onnx2tf" not in owner_source


def test_rank4_channelwise_broadcast_constant_repair_has_one_module_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper_name = (
        "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout"
    )
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout_pass"
    ]
    assert len(dispatches) == 1
    assert any(
        keyword.arg == "graph_index"
        for call in dispatches
        for keyword in call.keywords
    )

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3

    convergence = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_indexed_binary_layout_convergence"
    )
    convergence_calls = [
        node
        for node in ast.walk(convergence)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(convergence_calls) == 1
    assert any(
        keyword.arg == "graph_index"
        for call in convergence_calls
        for keyword in call.keywords
    )

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "binary_layout_adapter.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "repair_rank4_channelwise_broadcast_constants_to_runtime_layout"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    assert "ModelIRGraphIndex(model_ir)" in owner_function_source
    assert "graph_index.operator_indices_for_types(binary_ops)" in owner_function_source
    assert "graph_index.producer(name)" in owner_function_source
    assert "graph_index=graph_index" in owner_function_source
    assert "_set_operator_inputs(" in owner_function_source
    assert "lower_from_onnx2tf" not in owner_source


def test_convpool_output_passthrough_has_one_module_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper_name = "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_optimize_convpool_output_transpose_nhwc_passthrough_chains_pass"
    ]
    assert len(dispatches) == 1

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "convpool_output_passthrough_compat.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "optimize_convpool_output_transpose_nhwc_passthrough_chains"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    call_names = {
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_tensor_consumer_map" in call_names
    assert "_build_tensor_producer_map" in call_names
    assert "_set_operator_inputs" in call_names
    assert "_set_operator_outputs" in call_names
    assert "_replace_tensor_inputs" in call_names
    assert "_prune_unused_tensors" in call_names
    assert "del model_ir.operators[int(pre_idx)]" in owner_function_source
    prevalidation_offset = owner_function_source.index(
        "external_runtime_input_nhwc_shapes"
    )
    assert prevalidation_offset < owner_function_source.index(
        "channel_last_hint_names.add(str(pre_input_name))"
    )
    assert prevalidation_offset < owner_function_source.index(
        "_set_operator_inputs("
    )
    assert "lower_from_onnx2tf" not in owner_source

    focused_path = (
        REPO_ROOT
        / "tests"
        / "test_flatbuffer_direct_convpool_output_passthrough_layout.py"
    )
    assert focused_path.is_file()
    giant_source = (
        REPO_ROOT / "tests" / "test_tflite_builder_direct.py"
    ).read_text(encoding="utf-8")
    assert wrapper_name not in giant_source


def test_mean_hardsigmoid_muladd_has_one_module_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper_name = "_optimize_transpose_mean_hardsigmoid_muladd_chains"
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_optimize_transpose_mean_hardsigmoid_muladd_chains_pass"
    ]
    assert len(dispatches) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "mean_hardsigmoid_muladd_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "optimize_transpose_mean_hardsigmoid_muladd_chains"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    call_names = {
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_tensor_consumer_map" in call_names
    assert "_build_tensor_producer_map" in call_names
    assert "_read_const_ints_from_tensor" in call_names
    assert "_write_const_ints_to_tensor" in call_names
    assert "_replace_operator_input_at" in call_names
    assert "_prune_unused_tensors" in call_names
    assert "model_ir.operators.insert(" in owner_function_source
    assert "del model_ir.operators[int(remove_idx)]" in owner_function_source
    axes_validation = owner_function_source.index(
        "rank = len(list(q0_raw_tensor.shape))"
    )
    public_output_guard = owner_function_source.index(
        "if add0_out_name in model_ir.outputs:"
    )
    axes_write = owner_function_source.index(
        "_write_const_ints_to_tensor(\n"
        "                mean_axes_tensor,"
    )
    first_input_mutation = owner_function_source.index("_set_operator_inputs(")
    first_metadata_mutation = owner_function_source.index(
        "dq0_out_tensor.shape ="
    )
    assert public_output_guard < axes_validation
    assert axes_validation < axes_write < first_input_mutation
    assert axes_write < first_metadata_mutation
    assert "lower_from_onnx2tf" not in owner_source

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1

    focused_path = (
        REPO_ROOT
        / "tests"
        / "test_flatbuffer_direct_mean_hardsigmoid_muladd_layout.py"
    )
    assert focused_path.is_file()
    giant_source = (
        REPO_ROOT / "tests" / "test_tflite_builder_direct.py"
    ).read_text(encoding="utf-8")
    assert wrapper_name not in giant_source


def test_qlinear_concat_conv_has_one_module_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    wrapper_name = "_optimize_nhwc_propagation_qlinear_concat_conv"
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_optimize_nhwc_propagation_qlinear_concat_conv_pass"
    ]
    assert len(dispatches) == 1

    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "qlinear_concat_conv_compat.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "optimize_nhwc_propagation_qlinear_concat_conv"
    )
    owner_function_source = ast.get_source_segment(owner_source, owner)
    assert owner_function_source is not None
    call_names = {
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_tensor_consumer_map" in call_names
    assert "_build_tensor_producer_map" in call_names
    assert "_read_transpose_perm" in call_names
    assert "_set_operator_inputs" in call_names
    assert "_replace_tensor_inputs" in call_names
    assert "_prune_unused_tensors" in call_names
    assert "del model_ir.operators[int(remove_idx)]" in owner_function_source
    required_output_validation = owner_function_source.index(
        "if concat_out_tensor is None or q_out_tensor is None:"
    )
    public_pending_tensor_guard = owner_function_source.index(
        "str(tensor_name) in model_outputs\n"
        "                for tensor_name in pending_tensor_shape_updates"
    )
    first_input_mutation = owner_function_source.index(
        "_set_operator_inputs("
    )
    assert public_pending_tensor_guard < required_output_validation
    assert required_output_validation < first_input_mutation
    assert "lower_from_onnx2tf" not in owner_source

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1

    focused_path = (
        REPO_ROOT / "tests" / "test_flatbuffer_direct_qlinear_layout.py"
    )
    assert focused_path.is_file()
    giant_source = (
        REPO_ROOT / "tests" / "test_tflite_builder_direct.py"
    ).read_text(encoding="utf-8")
    assert wrapper_name not in giant_source


def test_dynamic_range_quantization_uses_differential_graph_index() -> None:
    quantization_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "quantization.py"
    )
    source = quantization_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_dynamic_range_quantized_model_ir"
    )
    function_source = ast.get_source_segment(source, function_node)
    assert function_source is not None
    assert "clone.operators =" not in function_source
    assert "ModelIRGraphIndex(clone)" in function_source
    assert "require_graph_index()" in function_source
    assert "active_index.insert_operator(" in function_source
    assert "active_index.replace_operator_inputs(" in function_source


def test_quantization_identity_elision_uses_batch_graph_index_removal() -> None:
    quantization_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "quantization.py"
    )
    source = quantization_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_elide_identity_operators"
    )
    function_source = ast.get_source_segment(source, function_node)
    assert function_source is not None
    assert "model_ir.operators =" not in function_source
    assert "ModelIRGraphIndex(model_ir)" in function_source
    assert "graph_index.replace_operator_inputs(" in function_source
    assert "graph_index.replace_operator_outputs(" in function_source
    assert "graph_index.remove_operators(" in function_source


def test_strict_integer_boundary_ops_use_differential_graph_index() -> None:
    quantization_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "quantization.py"
    )
    source = quantization_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_strict_full_integer_model_ir"
    )
    function_source = ast.get_source_segment(source, function_node)
    assert function_source is not None
    assert "clone.operators =" not in function_source
    assert "ModelIRGraphIndex(clone)" in function_source
    assert "require_graph_index()" in function_source
    assert "active_index.replace_operator_inputs(" in function_source
    assert "active_index.replace_operator_outputs(" in function_source
    assert "active_index.insert_operator(" in function_source
    assert "active_index.append_operator(" in function_source


def test_model_writer_reuses_shared_dead_operator_pruning() -> None:
    writer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "model_writer.py"
    )
    source = writer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prune_dead_operators_in_place"
    )
    function_source = ast.get_source_segment(source, function_node)
    assert function_source is not None
    assert "prune_dead_operators(model_ir, prune_tensors=False)" in function_source
    assert "keep_flags" not in function_source
    assert "model_ir.operators =" not in source


def test_split_rewrite_builder_owns_append_only_operator_stream() -> None:
    split_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "split_planner.py"
    )
    source = split_path.read_text(encoding="utf-8")
    assert "self.model_ir = copy.deepcopy(model_ir)" not in source
    assert "builder.model_ir.operators = rewritten_ops" not in source
    assert source.count("rewritten_ops = builder.model_ir.operators") == 3
    assert "operators=[]" in source
    crop_tree = ast.parse(source)
    crop_node = next(
        node
        for node in crop_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "crop_model_ir_by_boundary_tensors"
    )
    crop_source = ast.get_source_segment(source, crop_node)
    assert crop_source is not None
    assert "clone_operator_ir(" in crop_source
    assert "options=copy.deepcopy(dict(op.options))" in crop_source
    assert "axis_semantics=" in crop_source
    assert "onnx_node_name=" not in crop_source
    assert "onnx_op_type=" not in crop_source
    assert "kept_output_tensors" not in crop_source
    assert "set(str(name) for name in model_ir.inputs)" not in crop_source


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
    assert "run_boundary_input_batchmatmul_cleanup" in lowerer_names
    assert "run_boundary_input_normalization_cleanup" in lowerer_names

    graph_helpers = {
        "_broadcast_static_shapes",
        "_build_tensor_consumer_map",
        "_invert_perm",
        "_is_scalar_like_tensor",
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
        "_optimize_transpose_slice_muladd_conv_mergeadd_strict",
        "_optimize_transpose_slice_muladd_mergeadd_posttranspose_strict",
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
    assert "run_channel_slice_merge_layout_cleanup" in lowerer_names

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
    lowerer_names = {
        node.id
        for node in ast.walk(ast.parse(lowering_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Name)
    }
    assert "run_input_unary_passthrough_cleanup" in lowerer_names
    assert "run_hard_activation_passthrough_cleanup" in lowerer_names


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
        "_optimize_consecutive_reshape_passthrough_chains",
        "_optimize_fold_consecutive_mul_constants_chains",
        "_optimize_fuse_pseudo_leakyrelu_chains",
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
    pass_functions = _functions(pass_path)
    assert function_names <= set(pass_functions)
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names
    pseudo_leakyrelu = pass_functions[
        "_optimize_fuse_pseudo_leakyrelu_chains"
    ]
    pseudo_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(pseudo_leakyrelu)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in pseudo_calls
    assert "_build_tensor_producer_map" not in pseudo_calls
    assert "ModelIRGraphIndex" in pseudo_calls
    assert "operator_indices" in pseudo_calls
    assert "producer" in pseudo_calls
    assert "consumer_indices" in pseudo_calls
    assert "replace_operator_type" in pseudo_calls
    assert "replace_operator_inputs" in pseudo_calls
    assert "remove_operators" in pseudo_calls
    assert "_prune_unused_tensors" in pseudo_calls
    lowerer_names = {
        node.id
        for node in ast.walk(
            ast.parse(lowering_path.read_text(encoding="utf-8"))
        )
        if isinstance(node, ast.Name)
    }
    assert "run_clamp_cleanup" in lowerer_names
    assert "run_consecutive_reshape_cleanup" in lowerer_names
    assert "run_squeeze_reshape_identity_cleanup" in lowerer_names


def test_layout_transpose_cleanup_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "layout_transpose.py"
    )
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_is_identity_perm",
        "_is_inverse_perm",
        "_is_symmetric_terminal_binary_bridge_candidate",
        "_optimize_layout_transpose_chains",
        "_optimize_trailing_output_transpose_passthrough_chains",
        "_optimize_transpose_gather_transpose_axis_remap_nhwc_chains",
        "_optimize_transpose_gather_transpose_nhwc_channel_chains",
        "_optimize_transpose_unary_binary_full_post_fanout_bridges",
        "_optimize_transpose_unary_fanout_inverse_post_bridges",
        "_optimize_transpose_unary_passthrough_chains",
    } <= pass_functions

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(
            lowering_functions["_optimize_layout_transpose_chains"]
        )
        if isinstance(node, ast.Name)
    }
    assert "_optimize_layout_transpose_chains_pass" in wrapper_names
    gather_wrapper_names = {
        node.id
        for node in ast.walk(
            lowering_functions[
                "_optimize_transpose_gather_transpose_axis_remap_nhwc_chains"
            ]
        )
        if isinstance(node, ast.Name)
    }
    assert (
        "_optimize_transpose_gather_transpose_axis_remap_nhwc_chains_pass"
        in gather_wrapper_names
    )
    gather_fanout_wrapper_names = {
        node.id
        for node in ast.walk(
            lowering_functions[
                "_optimize_transpose_gather_transpose_nhwc_channel_chains"
            ]
        )
        if isinstance(node, ast.Name)
    }
    assert (
        "_optimize_transpose_gather_transpose_nhwc_channel_chains_pass"
        in gather_fanout_wrapper_names
    )
    unary_wrapper_names = {
        node.id
        for node in ast.walk(
            lowering_functions["_optimize_transpose_unary_passthrough_chains"]
        )
        if isinstance(node, ast.Name)
    }
    assert "_optimize_transpose_unary_passthrough_chains_pass" in unary_wrapper_names
    fanout_wrapper_names = {
        node.id
        for node in ast.walk(
            lowering_functions[
                "_optimize_transpose_unary_fanout_inverse_post_bridges"
            ]
        )
        if isinstance(node, ast.Name)
    }
    assert (
        "_optimize_transpose_unary_fanout_inverse_post_bridges_pass"
        in fanout_wrapper_names
    )
    binary_fanout_wrapper_names = {
        node.id
        for node in ast.walk(
            lowering_functions[
                "_optimize_transpose_unary_binary_full_post_fanout_bridges"
            ]
        )
        if isinstance(node, ast.Name)
    }
    assert (
        "_optimize_transpose_unary_binary_full_post_fanout_bridges_pass"
        in binary_fanout_wrapper_names
    )
    trailing_output_wrapper_names = {
        node.id
        for node in ast.walk(
            lowering_functions[
                "_optimize_trailing_output_transpose_passthrough_chains"
            ]
        )
        if isinstance(node, ast.Name)
    }
    assert (
        "_optimize_trailing_output_transpose_passthrough_chains_pass"
        in trailing_output_wrapper_names
    )
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.layout_transpose"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        "_is_identity_perm",
        "_is_inverse_perm",
        "_optimize_layout_transpose_chains",
        "_optimize_trailing_output_transpose_passthrough_chains",
        "_optimize_transpose_gather_transpose_axis_remap_nhwc_chains",
        "_optimize_transpose_gather_transpose_nhwc_channel_chains",
        "_optimize_transpose_unary_binary_full_post_fanout_bridges",
        "_optimize_transpose_unary_fanout_inverse_post_bridges",
        "_optimize_transpose_unary_passthrough_chains",
        "run_layout_transpose_cleanup",
        "run_trailing_output_transpose_cleanup",
        "run_transpose_gather_axis_cleanup",
        "run_transpose_gather_channel_fanout_cleanup",
        "run_transpose_unary_binary_fanout_bridge_cleanup",
        "run_transpose_unary_fanout_bridge_cleanup",
        "run_transpose_unary_passthrough_cleanup",
    }


def test_nchw_channel_shuffle_cleanup_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "channel_shuffle.py"
    )
    function_names = {
        "_optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather",
        "_optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains",
        "_optimize_shufflenet_transpose_shuffle_chains",
        "_repair_nchw_channel_shuffle_concat_gathers",
    }
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_names <= set(pass_functions)
    for function_name in function_names:
        function = pass_functions[function_name]
        referenced_names = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
        }
        assert "_build_tensor_consumer_map" not in referenced_names
        assert "_build_tensor_producer_map" not in referenced_names
        assert not any(
            isinstance(node, ast.Delete) for node in ast.walk(function)
        )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.channel_shuffle"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == function_names | {
        "run_nchw_channel_shuffle_cleanup",
        "run_nhwc_channel_shuffle_cleanup",
        "run_stale_nchw_channel_shuffle_repair",
        "run_two_way_channel_shuffle_cleanup",
    }


def test_mean_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "mean_layout.py"
    )
    function_names = {
        "_optimize_transpose_mean_prepost_nhwc_passthrough_chains",
        "_optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains",
    }

    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    assert function_names <= {
        node.name
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names

    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.mean_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == function_names | {
        "run_mean_mul_add_conv_layout_cleanup",
        "run_transpose_mean_passthrough_cleanup",
    }


def test_layernorm_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "layernorm_layout.py"
    )
    function_names = {
        "_optimize_transpose_layernorm_stats_nhwc_propagation_chains",
        "_optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains",
    }

    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    assert function_names <= {
        node.name
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names

    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.layernorm_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == function_names | {
        "run_layernorm_statistics_layout_cleanup",
    }


def test_terminal_mean_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_mean_layout.py"
    )
    function_name = "_optimize_transpose_pre_unary_mean_terminal_nhwc_chains"

    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names

    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.terminal_mean_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_terminal_mean_layout_cleanup",
    }


def test_se_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "se_layout.py"
    )
    function_names = {
        "_optimize_transpose_se_conv_mul_prepost_nhwc_chains",
        "_optimize_transpose_se_fc_mul_prepost_nhwc_chains",
    }

    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_names <= set(pass_functions)
    conv_function = pass_functions[
        "_optimize_transpose_se_conv_mul_prepost_nhwc_chains"
    ]
    conv_names = {
        node.id
        for node in ast.walk(conv_function)
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in conv_names
    assert "_build_tensor_producer_map" not in conv_names
    assert not any(
        isinstance(node, ast.Delete) for node in ast.walk(conv_function)
    )
    fc_function = pass_functions[
        "_optimize_transpose_se_fc_mul_prepost_nhwc_chains"
    ]
    fc_names = {
        node.id for node in ast.walk(fc_function) if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in fc_names
    assert "_build_tensor_producer_map" not in fc_names
    assert not any(
        isinstance(node, ast.Delete) for node in ast.walk(fc_function)
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names

    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.se_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == function_names | {
        "run_se_conv_layout_cleanup",
        "run_se_fc_layout_cleanup",
    }


def test_elementwise_gate_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "elementwise_gate_layout.py"
    )
    function_names = {
        "_optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains",
        "_optimize_transpose_weighted_add_swish_prepost_nhwc_chains",
        "_optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains",
        "_optimize_transpose_logistic_muladd_prepost_nhwc_chains",
    }
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_names <= set(pass_functions)
    for function_name in function_names:
        function = pass_functions[function_name]
        referenced_names = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
        }
        assert "_build_tensor_consumer_map" not in referenced_names
        assert "_build_tensor_producer_map" not in referenced_names
        assert not any(
            isinstance(node, ast.Delete) for node in ast.walk(function)
        )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names

    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.elementwise_gate_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == function_names | {
        "run_elementwise_gate_layout_cleanup",
    }


def test_multi_branch_gate_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "multi_branch_gate_layout.py"
    )
    function_name = (
        "_optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains"
    )
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.multi_branch_gate_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_multi_branch_gate_layout_cleanup",
    }


def test_complementary_gate_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "dual_postconv_gate_layout.py"
    )
    indexed_function_name = (
        "_optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains"
    )
    function_names = {
        indexed_function_name,
        "_optimize_transpose_logistic_sub_mul_postadd_nhwc_chains",
    }
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_names <= set(pass_functions)
    for function_name in function_names:
        referenced_names = {
            node.id
            for node in ast.walk(pass_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert "_build_tensor_consumer_map" not in referenced_names
        assert "_build_tensor_producer_map" not in referenced_names
        assert not any(
            isinstance(node, ast.Delete)
            for node in ast.walk(pass_functions[function_name])
        )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.dual_postconv_gate_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == function_names | {
        "run_dual_postconv_gate_layout_cleanup",
    }


def test_ndhwc_gate_layout_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "ndhwc_gate_layout.py"
    )
    function_names = {
        "_optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains",
        "_optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains",
    }
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_names <= set(pass_functions)
    for function_name in function_names:
        referenced_names = {
            node.id
            for node in ast.walk(pass_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert "_build_tensor_consumer_map" not in referenced_names
        assert "_build_tensor_producer_map" not in referenced_names
        assert not any(
            isinstance(node, ast.Delete)
            for node in ast.walk(pass_functions[function_name])
        )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.ndhwc_gate_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == function_names | {
        "run_ndhwc_gate_layout_cleanup",
    }


def test_cost_volume_scatter_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "cost_volume_scatter_layout.py"
    )
    function_name = "_optimize_transpose_cost_volume_scatter_ndhwc_chains"
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.cost_volume_scatter_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_cost_volume_scatter_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_cost_volume_scatter_layout_cleanup"
    ]
    assert len(runner_calls) == 2


def test_add_concat_suffix_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "add_concat_suffix_layout.py"
    )
    function_name = "_optimize_transpose_add_concat_const_suffix_nhwc_chains"
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.add_concat_suffix_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_add_concat_suffix_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_add_concat_suffix_layout_cleanup"
    ]
    assert len(runner_calls) == 1


def test_dual_mul_concat_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "dual_mul_concat_layout.py"
    )
    function_name = "_optimize_transpose_dual_mul_concat_prepost_nhwc_chains"
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.dual_mul_concat_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_dual_mul_concat_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_dual_mul_concat_layout_cleanup"
    ]
    assert len(runner_calls) == 2


def test_axis3_const_concat_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "axis3_const_concat_layout.py"
    )
    function_name = "_optimize_transpose_axis3_const_concat_bridge_nhwc_chains"
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.axis3_const_concat_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_axis3_const_concat_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_axis3_const_concat_layout_cleanup"
    ]
    assert len(runner_calls) == 1


def test_dequant_concat_quantize_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "dequant_concat_quantize_layout.py"
    )
    function_name = (
        "_optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    )
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.dequant_concat_quantize_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_dequant_concat_quantize_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_dequant_concat_quantize_layout_cleanup"
    ]
    assert len(runner_calls) == 2


def test_concat_unary_conv_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "concat_unary_conv_layout.py"
    )
    function_name = "_optimize_transpose_concat_unary_fanout_conv_nhwc_chains"
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions

    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.concat_unary_conv_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_concat_unary_conv_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_concat_unary_conv_layout_cleanup"
    ]
    assert len(runner_calls) == 2


def test_spp_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "spp_layout.py"
    )
    function_name = (
        "_optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
    )
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.spp_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_spp_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_spp_layout_cleanup"
    ]
    assert len(runner_calls) == 4


def test_ndhwc_pre_concat_layout_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "ndhwc_concat_layout.py"
    )
    function_name = "_optimize_transpose_pre_concat_ndhwc_chains"
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    pass_functions = {
        node.name: node
        for node in pass_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in pass_functions
    referenced_names = {
        node.id
        for node in ast.walk(pass_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert "_build_tensor_consumer_map" not in referenced_names
    assert "_build_tensor_producer_map" not in referenced_names
    assert not any(
        isinstance(node, ast.Delete)
        for node in ast.walk(pass_functions[function_name])
    )

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    lowering_functions = {
        node.name: node
        for node in lowering_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    imports = [
        node
        for node in lowering_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.ndhwc_concat_layout"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        function_name,
        "run_ndhwc_concat_layout_cleanup",
    }
    production_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function_name
    ]
    assert len(production_calls) == 0
    runner_calls = [
        call
        for call in ast.walk(lowering_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_ndhwc_concat_layout_cleanup"
    ]
    assert len(runner_calls) == 2


def test_ordered_model_ir_runner_calls_record_session_diagnostics() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    runner_names = {
        "run_add_concat_suffix_layout_cleanup",
        "run_axis3_const_concat_layout_cleanup",
        "run_boundary_input_layout_cleanup",
        "run_boundary_input_batchmatmul_cleanup",
        "run_boundary_input_normalization_cleanup",
        "run_channel_slice_merge_layout_cleanup",
        "run_clamp_cleanup",
        "run_consecutive_reshape_cleanup",
        "run_constant_input_fold_cleanup",
        "run_consecutive_mul_constants_cleanup",
        "run_concat_unary_conv_layout_cleanup",
        "run_conv_attention_layout_cleanup",
        "run_cost_volume_scatter_layout_cleanup",
        "run_dequant_concat_quantize_layout_cleanup",
        "run_duplicate_fanout_cleanup",
        "run_dual_mul_concat_layout_cleanup",
        "run_dual_postconv_gate_layout_cleanup",
        "run_elementwise_gate_layout_cleanup",
        "run_flatten_concat_reshape_cleanup",
        "run_mixed_attention_layout_cleanup",
        "run_multi_branch_gate_layout_cleanup",
        "run_mean_mul_add_conv_layout_cleanup",
        "run_nchw_channel_shuffle_cleanup",
        "run_nhwc_channel_shuffle_cleanup",
        "run_ndhwc_concat_layout_cleanup",
        "run_ndhwc_gate_layout_cleanup",
        "run_maximum_zero_relu_cleanup",
        "run_qkv_attention_bridge_cleanup",
        "run_qkv_attention_prefix_cleanup",
        "run_quantized_prelu_cleanup",
        "run_quantized_reshape_cleanup",
        "run_pad_layout_cleanup",
        "run_pad_mul_layout_cleanup",
        "run_normalization_pad_layout_cleanup",
        "run_input_unary_passthrough_cleanup",
        "run_layout_transpose_cleanup",
        "run_layernorm_statistics_layout_cleanup",
        "run_trailing_output_transpose_cleanup",
        "run_hard_activation_passthrough_cleanup",
        "run_redundant_cast_cleanup",
        "run_se_conv_layout_cleanup",
        "run_se_fc_layout_cleanup",
        "run_squeeze_reshape_identity_cleanup",
        "run_stale_nchw_channel_shuffle_repair",
        "run_singleton_maxpool_layout_cleanup",
        "run_singleton_channel_transpose_cleanup",
        "run_singleton_reshape_layout_cleanup",
        "run_singleton_spatial_reshape_cleanup",
        "run_spp_layout_cleanup",
        "run_terminal_quantize_dequantize_cleanup",
        "run_terminal_mean_layout_cleanup",
        "run_two_way_channel_shuffle_cleanup",
        "run_transpose_gather_axis_cleanup",
        "run_transpose_gather_channel_fanout_cleanup",
        "run_transpose_unary_binary_fanout_bridge_cleanup",
        "run_transpose_unary_fanout_bridge_cleanup",
        "run_transpose_unary_passthrough_cleanup",
        "run_transpose_mean_passthrough_cleanup",
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
    assert len(calls) == 118
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

    hard_activation_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_hard_activation_passthrough_cleanup"
    ]
    assert len(hard_activation_calls) == 2
    reverse_calls = [
        call
        for call in hard_activation_calls
        if any(
            keyword.arg == "reverse_hardsigmoid_order"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    ]
    assert len(reverse_calls) == 1
    assert any(
        keyword.arg == "include_hardswish"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in reverse_calls[0].keywords
    )

    reshape_only_duplicate_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_duplicate_fanout_cleanup"
        and any(
            keyword.arg == "include_transpose"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
    ]
    assert len(reshape_only_duplicate_calls) == 2

    boundary_batchmatmul_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_boundary_input_batchmatmul_cleanup"
    ]
    assert len(boundary_batchmatmul_calls) == 1

    pad_mul_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_pad_mul_layout_cleanup"
    ]
    assert len(pad_mul_calls) == 1

    channel_slice_merge_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_channel_slice_merge_layout_cleanup"
    ]
    assert len(channel_slice_merge_calls) == 1

    boundary_normalization_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_boundary_input_normalization_cleanup"
    ]
    assert len(boundary_normalization_calls) == 2

    quantized_prelu_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_quantized_prelu_cleanup"
    ]
    assert len(quantized_prelu_calls) == 2

    quantized_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_quantized_reshape_cleanup"
    ]
    assert len(quantized_reshape_calls) == 2

    singleton_maxpool_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_maxpool_layout_cleanup"
    ]
    assert len(singleton_maxpool_calls) == 2

    singleton_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_reshape_layout_cleanup"
    ]
    assert len(singleton_reshape_calls) == 1

    consecutive_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_consecutive_reshape_cleanup"
    ]
    assert len(consecutive_reshape_calls) == 4

    flatten_concat_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_flatten_concat_reshape_cleanup"
    ]
    assert len(flatten_concat_reshape_calls) == 1

    singleton_spatial_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_spatial_reshape_cleanup"
    ]
    assert len(singleton_spatial_reshape_calls) == 1

    singleton_channel_transpose_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_channel_transpose_cleanup"
    ]
    assert len(singleton_channel_transpose_calls) == 2

    layout_transpose_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_layout_transpose_cleanup"
    ]
    assert len(layout_transpose_calls) == 12

    transpose_gather_axis_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_gather_axis_cleanup"
    ]
    assert len(transpose_gather_axis_calls) == 3

    transpose_gather_channel_fanout_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_gather_channel_fanout_cleanup"
    ]
    assert len(transpose_gather_channel_fanout_calls) == 3

    transpose_unary_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_unary_passthrough_cleanup"
    ]
    assert len(transpose_unary_calls) == 3

    transpose_unary_fanout_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_unary_fanout_bridge_cleanup"
    ]
    assert len(transpose_unary_fanout_calls) == 3

    transpose_unary_binary_fanout_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_unary_binary_fanout_bridge_cleanup"
    ]
    assert len(transpose_unary_binary_fanout_calls) == 2

    trailing_output_transpose_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_trailing_output_transpose_cleanup"
    ]
    assert len(trailing_output_transpose_calls) == 1

    nchw_channel_shuffle_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_nchw_channel_shuffle_cleanup"
    ]
    assert len(nchw_channel_shuffle_calls) == 1

    nhwc_channel_shuffle_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_nhwc_channel_shuffle_cleanup"
    ]
    assert len(nhwc_channel_shuffle_calls) == 1

    two_way_channel_shuffle_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_two_way_channel_shuffle_cleanup"
    ]
    assert len(two_way_channel_shuffle_calls) == 1

    stale_nchw_channel_shuffle_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_stale_nchw_channel_shuffle_repair"
    ]
    assert len(stale_nchw_channel_shuffle_calls) == 1

    transpose_mean_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_mean_passthrough_cleanup"
    ]
    assert len(transpose_mean_calls) == 1

    mean_mul_add_conv_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_mean_mul_add_conv_layout_cleanup"
    ]
    assert len(mean_mul_add_conv_calls) == 2

    layernorm_statistics_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_layernorm_statistics_layout_cleanup"
    ]
    assert len(layernorm_statistics_calls) == 2

    terminal_mean_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_terminal_mean_layout_cleanup"
    ]
    assert len(terminal_mean_calls) == 1

    se_conv_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_se_conv_layout_cleanup"
    ]
    assert len(se_conv_calls) == 1

    se_fc_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_se_fc_layout_cleanup"
    ]
    assert len(se_fc_calls) == 3

    elementwise_gate_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_elementwise_gate_layout_cleanup"
    ]
    assert len(elementwise_gate_calls) == 1

    multi_branch_gate_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_multi_branch_gate_layout_cleanup"
    ]
    assert len(multi_branch_gate_calls) == 1

    dual_postconv_gate_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_dual_postconv_gate_layout_cleanup"
    ]
    assert len(dual_postconv_gate_calls) == 1

    ndhwc_gate_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_ndhwc_gate_layout_cleanup"
    ]
    assert len(ndhwc_gate_calls) == 2

    cost_volume_scatter_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_cost_volume_scatter_layout_cleanup"
    ]
    assert len(cost_volume_scatter_calls) == 2

    add_concat_suffix_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_add_concat_suffix_layout_cleanup"
    ]
    assert len(add_concat_suffix_calls) == 1

    dual_mul_concat_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_dual_mul_concat_layout_cleanup"
    ]
    assert len(dual_mul_concat_calls) == 2


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
        "_optimize_concat_pre_quantize_dequantize",
        "_optimize_terminal_quantize_dequantize",
        "_optimize_transpose_dequantize_mean_quantize_bridges",
        "_quantized_tensors_share_exact_grid",
        "_sanitize_terminal_transpose_before_dequantize",
    }

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    lowering_functions = _functions(lowering_path)
    pass_functions = _functions(pass_path)
    assert function_names <= set(pass_functions)
    for function_name in function_names:
        wrapper_names = {
            node.id
            for node in ast.walk(lowering_functions[function_name])
            if isinstance(node, ast.Name)
        }
        assert f"{function_name}_pass" in wrapper_names

    concat_owner = pass_functions["_optimize_concat_pre_quantize_dequantize"]
    concat_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(concat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in concat_calls
    assert "_build_tensor_producer_map" not in concat_calls
    assert "ModelIRGraphIndex" in concat_calls
    assert "operator_indices" in concat_calls
    assert "producer" in concat_calls
    assert "consumer_indices" in concat_calls
    assert "_set_operator_inputs" in concat_calls
    assert "_prune_unused_tensors" in concat_calls

    terminal_owner = pass_functions[
        "_sanitize_terminal_transpose_before_dequantize"
    ]
    terminal_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(terminal_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in terminal_calls
    assert "_build_tensor_producer_map" not in terminal_calls
    assert "ModelIRGraphIndex" in terminal_calls
    assert "operator_indices" in terminal_calls
    assert "producer" in terminal_calls
    assert "consumer_indices" in terminal_calls
    assert "_set_operator_inputs" in terminal_calls
    assert "_set_operator_outputs" in terminal_calls
    assert "_rename_tensor_globally" in terminal_calls
    assert "remove_operator" in terminal_calls
    assert "insert_operator" in terminal_calls

    mean_bridge_owner = pass_functions[
        "_optimize_transpose_dequantize_mean_quantize_bridges"
    ]
    mean_bridge_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(mean_bridge_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_build_tensor_consumer_map" not in mean_bridge_calls
    assert "_build_tensor_producer_map" not in mean_bridge_calls
    assert "ModelIRGraphIndex" in mean_bridge_calls
    assert "operator_indices" in mean_bridge_calls
    assert "consumer_indices" in mean_bridge_calls
    assert "_set_operator_inputs" in mean_bridge_calls
    assert "insert_operator" in mean_bridge_calls
    assert "remove_operator" in mean_bridge_calls
    assert "_prune_unused_tensors" in mean_bridge_calls
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
        "_optimize_attention_split_post_reshape_collapse_chains",
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
    assert "run_qkv_attention_prefix_cleanup" in lowerer_names
    pass_source = pass_path.read_text(encoding="utf-8")
    assert "_build_tensor_consumer_map" not in pass_source
    assert "_build_tensor_producer_map" not in pass_source
    assert "model_ir.operators.insert(" not in pass_source
    assert "del model_ir.operators" not in pass_source
    assert "graph_index.refresh()" not in pass_source


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
    lowerer_names = {
        node.id
        for node in ast.walk(ast.parse(lowering_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Name)
    }
    assert "run_pad_layout_cleanup" in lowerer_names
    assert "run_normalization_pad_layout_cleanup" in lowerer_names
    assert "run_pad_mul_layout_cleanup" in lowerer_names


def test_quantized_prelu_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_prelu.py"
    )
    function_names = {
        "_optimize_transpose_dequant_prelu_quantize_bridges",
        "_optimize_transpose_dequant_prelu_transpose_bridges",
        "_optimize_dequant_prelu_quantize_chains",
        "_optimize_dequant_prelu_depthwise_quantize_chains",
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
        for node in ast.walk(ast.parse(lowering_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Name)
    }
    assert "run_quantized_prelu_cleanup" in lowerer_names


def test_quantized_reshape_rewrite_has_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_reshape.py"
    )

    def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    function_name = "_optimize_dequant_reshape_quantize_chains"
    lowering_functions = _functions(lowering_path)
    assert function_name in _functions(pass_path)
    wrapper_names = {
        node.id
        for node in ast.walk(lowering_functions[function_name])
        if isinstance(node, ast.Name)
    }
    assert f"{function_name}_pass" in wrapper_names
    lowerer_names = {
        node.id
        for node in ast.walk(ast.parse(lowering_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Name)
    }
    assert "run_quantized_reshape_cleanup" in lowerer_names


def test_singleton_maxpool_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "singleton_maxpool_layout.py"
    )
    function_names = {
        "_optimize_singleton_layout_reshape_maxpool_binary_cast_chains",
        "_optimize_singleton_nms_maxpool_nhwc_chains",
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
        for node in ast.walk(ast.parse(lowering_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Name)
    }
    assert "run_singleton_maxpool_layout_cleanup" in lowerer_names


def test_singleton_reshape_rewrites_have_single_owner() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "singleton_reshape_layout.py"
    )
    function_names = {
        "_optimize_singleton_channel_layout_transpose_to_reshape",
        "_optimize_consecutive_inverse_singleton_layout_reshapes",
        "_optimize_flatten_concat_expanddims_to_nhwc_concat",
        "_optimize_singleton_layout_reshape_unary_passthrough_chains",
        "_optimize_singleton_reshape_concat_post_transpose_nhwc_chains",
        "_optimize_singleton_spatial_nhwc_transpose_reshape_flatten",
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
        for node in ast.walk(ast.parse(lowering_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Name)
    }
    assert "run_flatten_concat_reshape_cleanup" in lowerer_names
    assert "run_singleton_reshape_layout_cleanup" in lowerer_names
    assert "run_singleton_spatial_reshape_cleanup" in lowerer_names
    assert "run_singleton_channel_transpose_cleanup" in lowerer_names


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


def test_native_pytorch_emitters_have_single_owners() -> None:
    exporter_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    )
    emitter_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_emitters.py"
    )
    exporter_source = exporter_path.read_text(encoding="utf-8")
    emitter_source = emitter_path.read_text(encoding="utf-8")
    emitter_tree = ast.parse(emitter_source)
    emitter_function_nodes = {
        node.name: node
        for node in emitter_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    emitter_functions = set(emitter_function_nodes)

    assert "_emit_native_unary_op_for_codegen" in emitter_functions
    assert "_activation_lines_for_codegen" in emitter_functions
    assert "_emit_maybe_aligned_expr_for_codegen" in emitter_functions
    assert "_emit_module_output_expr_for_codegen" in emitter_functions
    assert "_emit_native_shape_transform_misc_op_for_codegen" in emitter_functions
    assert "_emit_native_binary_op_for_codegen" in emitter_functions
    assert "_emit_native_concat_op_for_codegen" in emitter_functions
    assert "_emit_native_recurrent_module_op_for_codegen" in emitter_functions
    assert (
        "_emit_native_fully_connected_module_op_for_codegen"
        in emitter_functions
    )
    assert "_emit_native_prelu_module_op_for_codegen" in emitter_functions
    assert (
        "_emit_native_transpose_conv2d_module_op_for_codegen"
        in emitter_functions
    )
    assert (
        "_emit_native_transpose_conv3d_module_op_for_codegen"
        in emitter_functions
    )
    assert "_emit_native_conv3d_module_op_for_codegen" in emitter_functions
    assert "_emit_native_conv2d_module_op_for_codegen" in emitter_functions
    assert "_emit_native_fused_module_op_for_codegen" in emitter_functions
    assert "_emit_native_direct_module_op_for_codegen" in emitter_functions
    assert (
        "_concat_channel_first_codegen_breaks_channel_last_consumers_for_codegen"
        in emitter_functions
    )
    assert "_emit_native_transpose_op_for_codegen" in emitter_functions
    assert "_DIRECT_CODEGEN_UNARY_EXPRESSIONS:" in emitter_source
    assert "def _emit_native_unary_op_for_codegen(" not in exporter_source
    assert "_emit_native_unary_op_for_codegen," in exporter_source
    assert "def _activation_lines_for_codegen(" not in exporter_source
    assert "_activation_lines_for_codegen," in exporter_source
    assert "def _emit_maybe_aligned_expr_for_codegen(" not in exporter_source
    assert "_emit_maybe_aligned_expr_for_codegen," in exporter_source
    assert "def _emit_module_output_expr_for_codegen(" not in exporter_source
    assert "_emit_module_output_expr_for_codegen," in exporter_source
    assert "_DIRECT_CODEGEN_UNARY_EXPRESSIONS:" not in exporter_source
    assert "_DIRECT_CODEGEN_UNARY_EXPRESSIONS," in exporter_source
    assert "_DIRECT_CODEGEN_BINARY_FUNCTIONS:" in emitter_source
    assert "_DIRECT_CODEGEN_BINARY_FUNCTIONS:" not in exporter_source
    assert "_DIRECT_CODEGEN_BINARY_FUNCTIONS," in exporter_source
    assert "def _emit_native_binary_op_for_codegen(" not in exporter_source
    assert "_emit_native_binary_op_for_codegen," in exporter_source
    assert (
        _NATIVE_CODEGEN_FUNCTION_SOURCE.count(
            "binary_output_target_shape_literal_fn="
        )
        == 1
    )
    ast.parse(_NATIVE_CODEGEN_FUNCTION_SOURCE)
    assert "def _emit_native_transpose_op_for_codegen(" not in exporter_source
    assert "_emit_native_transpose_op_for_codegen," in exporter_source
    assert "def _emit_native_concat_op_for_codegen(" not in exporter_source
    assert "_emit_native_concat_op_for_codegen," in exporter_source
    assert (
        "def _concat_channel_first_codegen_breaks_channel_last_consumers_for_codegen("
        not in exporter_source
    )
    assert "def _emit_native_direct_module_op_for_codegen(" not in exporter_source
    assert "_emit_native_direct_module_op_for_codegen," in exporter_source
    assert "_DIRECT_CODEGEN_MODULE_OP_TYPES:" in emitter_source
    assert "_DIRECT_CODEGEN_MODULE_OP_TYPES:" not in exporter_source
    assert "_DIRECT_CODEGEN_MODULE_OP_TYPES," in exporter_source
    direct_module_source = ast.get_source_segment(
        emitter_source,
        emitter_function_nodes["_emit_native_direct_module_op_for_codegen"],
    )
    assert direct_module_source is not None
    assert "_emit_native_recurrent_module_op_for_codegen(" in direct_module_source
    assert 'if op_type == "UNIDIRECTIONAL_SEQUENCE_RNN"' not in direct_module_source
    assert (
        'if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM"'
        not in direct_module_source
    )
    assert 'if op_type == "BIDIRECTIONAL_SEQUENCE_LSTM"' not in direct_module_source
    assert (
        "_emit_native_fully_connected_module_op_for_codegen("
        in direct_module_source
    )
    assert "_emit_native_prelu_module_op_for_codegen(" in direct_module_source
    assert 'if op_type == "FULLY_CONNECTED"' not in direct_module_source
    assert 'if op_type == "PRELU"' not in direct_module_source
    assert (
        "_emit_native_transpose_conv2d_module_op_for_codegen("
        in direct_module_source
    )
    assert (
        "_emit_native_transpose_conv3d_module_op_for_codegen("
        in direct_module_source
    )
    assert 'if op_type == "TRANSPOSE_CONV"' not in direct_module_source
    assert 'if op_type == "CONV_3D_TRANSPOSE"' not in direct_module_source
    assert "_emit_native_conv3d_module_op_for_codegen(" in direct_module_source
    assert 'op_type == "CONV_3D"' not in direct_module_source
    assert "_emit_native_conv2d_module_op_for_codegen(" in direct_module_source
    assert 'if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}' not in direct_module_source
    assert "_emit_native_fused_module_op_for_codegen(" in direct_module_source
    assert "if fused_module_spec is not None:" not in direct_module_source
    assert "forward_lines.append(" not in direct_module_source
    assert ".permute(" not in direct_module_source
    assert (
        "def _emit_native_shape_transform_misc_op_for_codegen("
        not in exporter_source
    )
    assert "_emit_native_shape_transform_misc_op_for_codegen," in exporter_source
    assert "def _constant_int_list(" not in exporter_source
    codegen_utils_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_codegen_utils.py"
    ).read_text(encoding="utf-8")
    assert "def _constant_int_list(" in codegen_utils_source
    assert "_constant_int_list," in emitter_source
    # The dynamically compiled legacy codegen body resolves helpers from the
    # exporter module globals, so its current owner must also be bound here.
    assert "_constant_int_list," in exporter_source
    assert "logical_layout_permutation," in exporter_source


def test_native_pytorch_stage_codegen_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    stage_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_codegen_stages.py"
    ).read_text(encoding="utf-8")

    assert "def _build_named_encoder_methods_composite(" in stage_source
    assert "def _build_named_encoder_methods_composite(" not in exporter_source
    assert "_build_named_encoder_methods_composite," in exporter_source
    assert "def _build_forward_stage_methods(" in stage_source
    assert "def _build_forward_stage_methods(" not in exporter_source
    assert "_build_forward_stage_methods," in exporter_source
    assert "def _fold_single_use_static_reshape_chains(" in stage_source
    assert "def _fold_single_use_static_reshape_chains(" not in exporter_source
    assert "_fold_single_use_static_reshape_chains," in exporter_source
    assert "def _build_named_encoder_methods(" not in exporter_source
    assert "import torch" not in stage_source


def test_native_pytorch_legacy_codegen_compatibility_bindings_are_present() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    layout_utils_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_layout_utils.py"
    ).read_text(encoding="utf-8")
    runtime_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_package_runtime.py"
    ).read_text(encoding="utf-8")

    for function_name in (
        "_compose_axis_permutations",
        "_perm_cf_to_cl",
    ):
        assert f"def {function_name}(" in layout_utils_source
        assert f"def {function_name}(" not in exporter_source
        assert f"{function_name}," in exporter_source
    assert "model = _get_generated_model_cls()(metadata=metadata)" in runtime_source
    assert "model = _GeneratedModel(metadata=metadata)" not in runtime_source


def test_torchscript_artifact_export_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    artifact_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_artifact_exporters.py"
    ).read_text(encoding="utf-8")
    support_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_export_support.py"
    ).read_text(encoding="utf-8")

    assert "def export_torchscript_from_generated_package(" in artifact_source
    assert (
        "def export_torchscript_from_generated_package("
        not in exporter_source
    )
    assert "export_torchscript_from_generated_package," in exporter_source
    assert "def _suppress_torch_onnx_optional_registration_warnings(" not in exporter_source
    assert (
        'logging.getLogger("torch.onnx._internal.exporter._registration")'
        in artifact_source
    )
    for helper_name in (
        "_build_metadata_payload",
        "_metadata_has_dynamic_public_inputs",
        "_generated_package_non_native_skip_reason",
        "_generated_package_torch_export_skip_reason",
        "_run_generated_package_export_child",
    ):
        assert f"def {helper_name}(" in support_source
        assert f"def {helper_name}(" not in exporter_source
        assert f"{helper_name}," in exporter_source
    for support_only_helper_name in (
        "_serializable_tensor_meta",
        "_serializable_value",
    ):
        assert f"def {support_only_helper_name}(" in support_source
        assert f"def {support_only_helper_name}(" not in exporter_source
        assert f"{support_only_helper_name}," not in exporter_source

    for source in (artifact_source, support_source):
        top_level_imports = {
            alias.name
            for node in ast.parse(source).body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "torch" not in top_level_imports


def test_backed_pytorch_package_exports_and_metadata_have_single_owners() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    artifact_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_artifact_exporters.py"
    ).read_text(encoding="utf-8")
    support_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_export_support.py"
    ).read_text(encoding="utf-8")

    def _functions(source: str) -> set[str]:
        return {
            node.name
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    exporter_functions = _functions(exporter_source)
    artifact_functions = _functions(artifact_source)
    support_functions = _functions(support_source)
    for function_name in (
        "export_pytorch_package_from_saved_model_artifact",
        "export_pytorch_package_from_tflite_artifact",
    ):
        assert function_name in artifact_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    for function_name in (
        "_build_saved_model_backed_metadata_payload",
        "_build_tflite_backed_metadata_payload",
    ):
        assert function_name in support_functions
        assert function_name not in exporter_functions


def test_dynamo_onnx_artifact_export_has_focused_owners() -> None:
    exporter_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    )
    exporter_source = exporter_path.read_text(encoding="utf-8")
    exporter_functions = {
        node.name: node
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    artifact_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_artifact_exporters.py"
    ).read_text(encoding="utf-8")
    onnx_support_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_onnx_artifact_support.py"
    ).read_text(encoding="utf-8")

    assert "def _export_dynamo_onnx_from_generated_package(" in artifact_source
    wrapper_source = ast.get_source_segment(
        exporter_source,
        exporter_functions["export_dynamo_onnx_from_generated_package"],
    )
    assert wrapper_source is not None
    assert "_export_dynamo_onnx_from_generated_package(" in wrapper_source
    assert "child_script" not in wrapper_source
    assert "_write_generated_package_export_metadata(" not in wrapper_source
    assert (
        "_temporarily_rewrite_generated_model_source_for_exported_program"
        in wrapper_source
    )
    assert "_reapply_post_export_final_model_repairs" in wrapper_source

    assert "def _sanitize_dynamo_exported_onnx_metadata(" in onnx_support_source
    assert "def _sanitize_dynamo_exported_onnx_metadata(" not in exporter_source
    assert "_sanitize_dynamo_exported_onnx_metadata," in exporter_source
    for helper_name in (
        "_onnx_model_uses_external_data",
        "_inspect_onnx_uses_external_data",
        "_restore_missing_onnx_output_shapes_from_package_metadata",
    ):
        assert f"def {helper_name}(" in onnx_support_source
        assert f"def {helper_name}(" not in exporter_source


def test_pytorch_onnx_boundary_inference_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    support_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_onnx_artifact_support.py"
    ).read_text(encoding="utf-8")

    def _functions(source: str) -> set[str]:
        return {
            node.name
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    exporter_functions = _functions(exporter_source)
    support_functions = _functions(support_source)
    for function_name in (
        "_infer_batchless_rank3_image_boundaries_from_onnx_graph",
        "_infer_public_layouts_from_onnx_graph",
        "_is_onnx_boundary_layout_passthrough_node",
        "_merge_reference_public_boundary_metadata",
        "_read_onnx_transpose_perm",
    ):
        assert function_name in support_functions
        assert function_name not in exporter_functions
    assert "_merge_reference_public_boundary_metadata," in exporter_source
def test_exported_program_child_script_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    artifact_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_artifact_exporters.py"
    ).read_text(encoding="utf-8")
    child_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_exported_program_child.py"
    ).read_text(encoding="utf-8")

    assert "_EXPORTED_PROGRAM_CHILD_SCRIPT =" in child_source
    assert "child_script = _EXPORTED_PROGRAM_CHILD_SCRIPT" in artifact_source
    assert "child_script = \"\"\"" not in exporter_source
    assert artifact_source.count("child_script = \"\"\"") == 2
    assert "_EXPORTED_PROGRAM_CHILD_SCRIPT," in artifact_source


def test_exported_program_artifact_host_has_focused_owner() -> None:
    exporter_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    )
    exporter_source = exporter_path.read_text(encoding="utf-8")
    exporter_functions = {
        node.name: node
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    artifact_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_artifact_exporters.py"
    ).read_text(encoding="utf-8")
    archive_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_exported_program_archive.py"
    ).read_text(encoding="utf-8")

    assert (
        "def _export_exported_program_from_generated_package("
        in artifact_source
    )
    wrapper_source = ast.get_source_segment(
        exporter_source,
        exporter_functions[
            "export_exported_program_from_generated_package"
        ],
    )
    assert wrapper_source is not None
    assert "_export_exported_program_from_generated_package(" in wrapper_source
    assert "child_script" not in wrapper_source
    assert "_write_generated_package_export_metadata(" not in wrapper_source
    for callback_name in (
        "_temporarily_rewrite_generated_model_source_for_exported_program",
        "_reapply_post_export_final_model_repairs",
    ):
        assert callback_name in wrapper_source
    assert "def _strip_stack_traces_from_exported_program_archive(" in (
        archive_source
    )
    assert "def _strip_stack_traces_from_exported_program_archive(" not in (
        exporter_source
    )
    assert "_strip_stack_traces_from_exported_program_archive(" in (
        artifact_source
    )
    assert (
        "def _fold_inverse_permute_round_trips_in_exported_program_archive("
        in archive_source
    )
    assert (
        "def _fold_inverse_permute_round_trips_in_exported_program_archive("
        not in exporter_source
    )
    assert (
        "_fold_inverse_permute_round_trips_in_exported_program_archive("
        in artifact_source
    )


def test_generated_pytorch_source_parsers_have_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    parser_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_source_parser.py"
    ).read_text(encoding="utf-8")
    fast_policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_fast_precanonicalize_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    parser_functions = {
        node.name
        for node in ast.parse(parser_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_parse_aligned_binary_assign_with_shape",
        "_parse_apply_resize_assign",
        "_parse_dynamic_apply_pool2d_assign",
        "_parse_local_response_norm_assign",
        "_parse_reduce_max_assign",
        "_parse_simple_return_identifier",
        "_parse_int_list_literal",
        "_strip_outer_parentheses",
        "_split_top_level_csv_exprs",
        "_parse_binary_add_args",
        "_parse_binary_mul_args",
        "_parse_binary_sub_args",
        "_parse_align_tensor_target_shape_expr",
        "_parse_simple_assignment_line_cached",
        "_parse_simple_assignment_line",
        "_parse_rank4_shape_literal",
        "_parse_apply_concat_inputs_axis_and_shape",
        "_parse_torch_cat_inputs_and_dim",
        "_normalize_permute_dims_expr",
        "_parse_channel_last_gather_slice_assign",
        "_parse_rank4_shape_expr",
        "_parse_apply_resize_input_size_shape_and_channel_last",
        "_parse_apply_pool2d_input_channel_last_and_is_max",
        "_parse_apply_pool2d_assign_with_shape",
        "_parse_tensor_split_assign",
        "_parse_apply_softmax_input_axis_and_shape",
        "_resolve_nhwc_to_nchw_bridge_source",
        "_parse_copy_call_expr",
        "_parse_align_tensor_target_shape_assign",
        "_parse_torch_permute_assign",
        "_parse_local_response_norm_input_expr",
        "_parse_apply_pool2d_input_expr",
        "_parse_apply_resize_input_and_channel_last",
        "_parse_apply_pool2d_input_and_channel_last",
        "_parse_apply_softmax_input_and_axis",
        "_parse_constant_pad_assign",
        "_parse_dynamic_binary_add_align_assign",
        "_parse_static_binary_add_align_assign",
        "_parse_align_binary_inputs_to_anchor_assign_with_shape",
        "_model_source_lines",
        "_any_line_matches",
        "_count_lines_matching",
        "_extract_prefixed_call_exprs",
    ):
        assert function_name in parser_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN =" in parser_source
    assert "_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN =" not in exporter_source
    assert "_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN," in exporter_source
    assert "import torch" not in parser_source
    assert "_parse_apply_resize_assign," in fast_policy_source
    assert "_parse_dynamic_binary_align_assign," in fast_policy_source
    assert "_parse_dynamic_binary_align_assign" in parser_functions
    assert "_parse_dynamic_binary_align_assign" not in exporter_functions
    for function_name in (
        "_parse_aligned_rank4_assign",
        "_parse_apply_softmax_assign",
        "_parse_permuted_conv_input_assign",
    ):
        assert function_name in parser_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," not in exporter_source
        assert f"{function_name}," in fast_policy_source
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_parse_apply_resize_assign"
        for node in ast.walk(ast.parse(fast_policy_source))
    )

    exporter_module = ast.parse(exporter_source)
    orchestrator = next(
        node
        for node in exporter_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_fast_precanonicalize_repairs"
    )
    loaded_names = {
        node.id
        for node in ast.walk(orchestrator)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in ast.walk(orchestrator):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assert target.id in loaded_names


def test_generated_pytorch_naming_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    naming_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_naming.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    naming_functions = {
        node.name
        for node in ast.parse(naming_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    moved_functions = (
        "_build_buffer_attr_name_map",
        "_build_tensor_var_name_map",
        "_canonical_codegen_name_for_codegen",
        "_collapse_generated_name_tokens",
        "_direct_codegen_module_attr_base",
        "_extract_generated_name_suffix_tokens",
        "_make_tensor_storage_name_map",
        "_make_unique_identifier",
        "_next_unique_attr_name_for_codegen",
        "_sanitize_python_identifier",
        "_shorten_generated_python_identifier",
        "_split_generated_name_piece",
    )
    for function_name in moved_functions:
        assert function_name in naming_functions
        assert function_name not in exporter_functions
    for imported_name in (
        "_build_buffer_attr_name_map",
        "_build_tensor_var_name_map",
        "_canonical_codegen_name_for_codegen",
        "_direct_codegen_module_attr_base",
        "_make_tensor_storage_name_map",
        "_make_unique_identifier",
        "_next_unique_attr_name_for_codegen",
        "_sanitize_python_identifier",
        "_shorten_generated_python_identifier",
    ):
        assert f"{imported_name}," in exporter_source
    assert "_direct_codegen_module_attr_name" not in exporter_functions
    for constant_name in (
        "_GENERATED_NAME_DROP_TOKENS",
        "_GENERATED_NAME_SUFFIX_PATTERNS",
        "_GENERATED_NAME_TOKEN_ALIASES",
        "_PYTORCH_LOCAL_NAME_MAX_LENGTH",
    ):
        assert constant_name in naming_source
        assert constant_name not in exporter_source
    assert "import torch" not in naming_source


def test_generated_pytorch_codegen_values_have_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    values_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_codegen_values.py"
    ).read_text(encoding="utf-8")
    constant_policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_constant_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    values_functions = {
        node.name
        for node in ast.parse(values_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    exporter_imported_functions = (
        "_conv_block_activation_config",
        "_conv_block_activation_config_from_fused_name",
        "_is_small_inline_constant_tensor",
        "_python_literal_for_constant_tensor",
        "_torch_dtype_literal",
    )
    constant_policy_imported_functions = (
        "_scalar_literal_for_constant_tensor",
        "_torch_pad_literal_for_constant_tensor",
    )
    for function_name in (
        *exporter_imported_functions,
        *constant_policy_imported_functions,
    ):
        assert function_name in values_functions
        assert function_name not in exporter_functions
    for function_name in exporter_imported_functions:
        assert f"{function_name}," in exporter_source
    for function_name in constant_policy_imported_functions:
        assert f"{function_name}," in constant_policy_source
    # The dynamically compiled legacy codegen body still resolves the pad
    # literal helper from the exporter module globals.
    assert "_torch_pad_literal_for_constant_tensor," in exporter_source
    assert "import torch" not in values_source


def test_generated_pytorch_indexing_codegen_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    indexing_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_indexing_codegen.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    indexing_functions = {
        node.name
        for node in ast.parse(indexing_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    moved_functions = (
        "_direct_dynamic_gather_expr",
        "_direct_gather_expr",
        "_direct_gather_reshape_expr",
        "_direct_slice_expr",
        "_direct_strided_slice_expr",
        "_direct_symbolic_strided_slice_expr",
        "_is_suffix_flatten_gather_reshape",
        "_reshape_is_plain_singleton_axis_drop",
        "_should_elide_crd_to_dcr_gather_for_depth_to_space",
    )
    for function_name in moved_functions:
        assert function_name in indexing_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in indexing_source


def test_native_pytorch_codegen_uses_shared_model_ir_graph_index() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    context_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "_pytorch_exporter_native_codegen_common.py"
    ).read_text(encoding="utf-8")
    exporter_tree = ast.parse(exporter_source)
    exporter_functions = {
        node.name: node
        for node in exporter_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_build_model_ir_producer_consumer_index" not in exporter_functions
    assert "_assemble_native_model_source" not in exporter_functions
    writer = exporter_functions["_write_native_model_file"]
    constructor_calls = [
        node
        for node in ast.walk(writer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRGraphIndex"
    ]
    assert len(constructor_calls) == 1
    context_class = next(
        node
        for node in ast.parse(context_source).body
        if isinstance(node, ast.ClassDef)
        and node.name == "_NativeModelFileWriterContext"
    )
    context_fields = {
        node.target.id
        for node in context_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    assert "graph_index" in context_fields
    assert "producer_index" not in context_fields
    assert "consumer_index" not in context_fields
    assert "def producer_index(" in context_source
    assert "def consumer_index(" in context_source
    collector_calls = [
        node
        for node in ast.walk(writer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_collect_feature_last_sequence_tensor_names"
    ]
    assert len(collector_calls) == 1
    assert any(
        keyword.arg == "graph_index"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "graph_index"
        for keyword in collector_calls[0].keywords
    )
    context_calls = [
        node
        for node in ast.walk(writer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_NativeModelFileWriterContext"
    ]
    assert len(context_calls) == 1
    assert isinstance(context_calls[0].args[-1], ast.Name)
    assert context_calls[0].args[-1].id == "graph_index"
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"producer_index", "consumer_index"}
        and node.func.attr in {"clear", "pop", "setdefault", "update"}
        for node in ast.walk(exporter_tree)
    )


def test_native_pytorch_state_dict_support_has_single_owner_and_lazy_torch() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    support_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_state_dict_support.py"
    ).read_text(encoding="utf-8")
    exporter_tree = ast.parse(exporter_source)
    support_tree = ast.parse(support_source)
    exporter_functions = {
        node.name
        for node in exporter_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    support_functions = {
        node.name: node
        for node in support_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    moved_functions = (
        "_build_native_generated_state_dict",
        "_import_generated_package_from_output",
        "_prepare_exported_state_tensor",
    )
    for function_name in moved_functions:
        assert function_name in support_functions
        assert function_name not in exporter_functions
    assert "_build_native_generated_state_dict," in exporter_source
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "torch" for alias in node.names)
        for node in support_tree.body
    )
    prepare_imports = [
        node
        for node in ast.walk(support_functions["_prepare_exported_state_tensor"])
        if isinstance(node, ast.Import)
        and any(alias.name == "torch" for alias in node.names)
    ]
    assert len(prepare_imports) == 1


def test_generated_pytorch_package_sources_have_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    package_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_package_sources.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    package_functions = {
        node.name
        for node in ast.parse(package_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    moved_functions = (
        "_build_native_runtime_source",
        "_patch_generated_runtime_pool2d_channel_last_recovery",
        "_write_generated_package_common_files",
        "_write_wrapper_model_file",
    )
    for function_name in moved_functions:
        assert function_name in package_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source


def test_pytorch_backed_package_selection_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    selection_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_package_selection.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    selection_functions = {
        node.name
        for node in ast.parse(selection_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_has_tflite_import_preferred_control_or_recurrent_ops",
        "_should_prefer_saved_model_backed_package",
        "_should_prefer_tflite_backed_package",
    ):
        assert function_name in selection_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    exporter_calls = [
        node.func.id
        for node in ast.walk(ast.parse(exporter_source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert exporter_calls.count("_should_prefer_saved_model_backed_package") == 1
    assert exporter_calls.count("_should_prefer_tflite_backed_package") == 2
    assert exporter_calls.count(
        "_has_tflite_import_preferred_control_or_recurrent_ops"
    ) == 1
    assert "model_op_types =" not in exporter_source
    assert "control_or_recurrent_ops =" not in exporter_source
    assert "import torch" not in selection_source


def test_pytorch_runtime_wrapper_export_has_single_owner_and_lazy_torch() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    wrapper_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_runtime_wrapper_exporter.py"
    ).read_text(encoding="utf-8")
    exporter_tree = ast.parse(exporter_source)
    wrapper_tree = ast.parse(wrapper_source)
    exporter_functions = {
        node.name
        for node in exporter_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper_functions = {
        node.name: node
        for node in wrapper_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    function_name = "_export_runtime_wrapper_package_from_model_ir"
    assert function_name in wrapper_functions
    assert function_name not in exporter_functions
    assert f"{function_name}," in exporter_source
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "torch" for alias in node.names)
        for node in wrapper_tree.body
    )
    lazy_imports = [
        node
        for node in ast.walk(wrapper_functions[function_name])
        if isinstance(node, ast.Import)
        and any(alias.name == "torch" for alias in node.names)
    ]
    assert len(lazy_imports) == 1


def test_pytorch_string_normalizer_export_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    string_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_string_normalizer_exporter.py"
    ).read_text(encoding="utf-8")

    def _functions(source: str) -> set[str]:
        return {
            node.name
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    exporter_functions = _functions(exporter_source)
    string_functions = _functions(string_source)
    for function_name in (
        "_extract_string_normalizer_config_from_onnx_graph",
        "export_pytorch_package_from_string_normalizer_onnx",
    ):
        assert function_name in string_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in string_source
    assert "tensorflow" not in string_source


def test_pytorch_capability_registry_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    capability_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_capabilities.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    capability_functions = {
        node.name
        for node in ast.parse(capability_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_can_emit_direct_module_call_for_codegen",
        "_ensure_direct_codegen_supported",
        "_ensure_native_export_supported_ops",
        "_ensure_no_custom_ops",
        "_ensure_supported_ops",
        "_is_direct_codegen_unsupported_error",
        "_is_channel_last_layout_for_codegen",
        "_supports_runtime_wrapper_model_ir",
        "get_supported_pytorch_kernel_op_types",
    ):
        assert function_name in capability_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    exporter_calls = [
        node.func.id
        for node in ast.walk(ast.parse(exporter_source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert exporter_calls.count("_ensure_native_export_supported_ops") == 2
    assert "_ensure_no_custom_ops" not in exporter_calls
    assert "_ensure_supported_ops" not in exporter_calls
    assert "_DIRECT_CODEGEN_SUPPORTED_OP_TYPES: Set[str] =" in capability_source
    assert "_DIRECT_CODEGEN_SUPPORTED_OP_TYPES: Set[str] =" not in exporter_source
    assert "_DIRECT_CODEGEN_SUPPORTED_OP_TYPES," in exporter_source
    assert "_RUNTIME_SUPPORTED_CUSTOM_CODES: Set[str] =" in capability_source
    assert "_RUNTIME_SUPPORTED_CUSTOM_CODES: Set[str] =" not in exporter_source
    assert "import torch" not in capability_source


def test_generated_pytorch_graph_source_rewrites_have_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    graph_rewrite_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_source_graph_rewrites.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    graph_rewrite_functions = {
        node.name
        for node in ast.parse(graph_rewrite_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_bridge_boundary_metadata_gather_nd_inputs",
        "_infer_gather_nd_shape_for_codegen",
    ):
        assert function_name in graph_rewrite_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in graph_rewrite_source


def test_generated_pytorch_graph_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_graph_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_base_target_shape_values_for_model_ir",
        "_channel_first_shape_for_tensor_for_codegen",
        "_channel_first_shape_values_for_model_ir",
        "_expected_channel_dim_for_tensor_for_codegen",
        "_gather_input_pre_permute_for_codegen",
        "_infer_effective_rank4_runtime_layout_for_codegen",
        "_is_sequential_single_input_graph_for_codegen",
        "_native_codegen_cache_bucket_for_model_ir",
        "_producer_op_for_model_ir",
        "_rank4_channel_first_shape_for_tensor_for_codegen",
        "_resize_target_shape_literal_for_model_ir",
        "_resolve_channel_first_named_tensor_shape_for_codegen",
        "_target_shape_literal_for_model_ir",
        "_target_shape_values_for_model_ir",
        "_tensor_shape_list_for_model_ir",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "_native_codegen_graph_index_for_model_ir" in policy_functions
    assert "_native_codegen_expected_channel_dim_cache_for_model_ir" in policy_functions
    assert "_to_channel_first_shape_for_model_ir" in policy_functions
    assert (
        "_expected_channel_dim_for_channel_last_named_tensor_for_codegen"
        in policy_functions
    )
    assert "_native_codegen_producer_lookup_for_model_ir" not in exporter_functions
    assert "_native_codegen_producer_lookup_for_model_ir" not in policy_functions
    assert 'cache_bucket["graph_index"] = context.graph_index' in exporter_source
    assert 'graph_index.operator_indices("CONV_2D")' in policy_source
    assert "import torch" not in policy_source


def test_generated_pytorch_concat_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_concat_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_can_fold_channel_last_alias_slice_consumer_for_codegen",
        "_can_keep_channel_first_slice_output_for_codegen",
        "_channel_first_concat_input_expr_for_codegen",
        "_is_valid_concat_axis_for_channel_first_shapes_for_codegen",
        "_resolve_concat_axis_for_channel_first_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_channel_first_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_channel_first_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_can_emit_channel_first_shape_preserving_unary_op_for_codegen",
        "_can_resolve_channel_first_expr_statically_for_codegen",
        "_channel_first_passthrough_input_expr_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_binary_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_binary_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_all_consumers_are_channel_first_binary_ops_for_codegen",
        "_binary_operand_expr_for_codegen",
        "_binary_output_target_shape_literal_for_codegen",
        "_binary_requires_runtime_alignment_for_codegen",
        "_binary_runtime_shape_passthrough_operand_for_codegen",
        "_can_emit_channel_first_binary_op_for_codegen",
        "_can_omit_materialized_channel_last_alias_recursive_for_codegen",
        "_channel_first_binary_input_expr_for_codegen",
        "_preferred_binary_alignment_anchor_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_layout_bridge_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_layout_bridge_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_fold_single_consumer_public_input_bridge_for_codegen",
        "_has_channel_last_consumer_hint_for_same_shape_transpose_for_codegen",
        "_is_batchless_rank3_public_output_transpose_for_codegen",
        "_match_single_consumer_layout_bridge_transpose_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_shape_expression_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_shape_expression_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_reconstruct_shape_list_expr_for_codegen",
        "_reconstruct_shape_scalar_expr_for_codegen",
        "_shape_tensor_length_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_recurrent_codegen_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_recurrent_codegen_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_require_constant_array_from_model_ir",
        "_sequence_lstm_bias_array_for_model_ir",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_fast_precanonicalize_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_fast_precanonicalize_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    exporter_classes = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, ast.ClassDef)
    }
    policy_classes = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, ast.ClassDef)
    }

    for function_name in (
        "_build_fast_precanonicalize_repair_context",
        "_convert_nchw_pad_to_nhwc_pad_values",
        "_convert_nhwc_pad_to_nchw_pad_values",
        "_fast_precanonicalize_expr_identifiers",
        "_fast_precanonicalize_has_channel_last_spatial_consumer",
        "_fast_precanonicalize_infer_consumer_layout",
        "_fast_precanonicalize_is_cf_like",
        "_fast_precanonicalize_is_nhwc_like",
        "_fast_precanonicalize_preferred_channel_count",
        "_fast_precanonicalize_resolve_alias",
        "_has_immediate_rank4_permute_source",
        "_infer_unique_channel_count_from_rank4_shape",
        "_repair_binary_alignment_layout",
        "_repair_binary_alignment_from_downstream_evidence",
        "_repair_aligned_scalar_binary_shape_at",
        "_repair_aligned_bn_constant_layout",
        "_repair_cf_pool_target_shape",
        "_repair_cf_pool_neighbor_layout_at",
        "_repair_cf_resize_from_input_and_bn_evidence",
        "_repair_cf_gather_slice_at",
        "_repair_cf_reduce_max_axis",
        "_repair_cf_resize_target_shape",
        "_repair_cf_softmax_axis",
        "_repair_concat_axis_from_input_layouts",
        "_repair_depth_to_space_gather_at",
        "_repair_dynamic_cf_binary_anchor_at",
        "_repair_dynamic_cf_binary_anchor_shapes",
        "_repair_dynamic_pool_layout_at",
        "_repair_nhwc_average_pool_binary_bridge",
        "_repair_nhwc_buffer_binary_alignment_at",
        "_repair_nhwc_pool_layout",
        "_repair_simple_alias_layout_at",
        "_repair_singleton_reshape_cf_binary_at",
        "_repair_split_axis_from_consumers",
        "_repair_terminal_classifier_tail_layout",
        "_propagate_cf_local_response_norm_output",
        "_propagate_cf_prelu_output",
        "_record_rewritten_static_shape",
        "_restore_channel_last_spatial_pool_chains",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "_FastPrecanonicalizeRepairContext" in policy_classes
    assert "_FastPrecanonicalizeRepairContext" not in exporter_classes
    assert "_FastPrecanonicalizeRepairContext," in exporter_source
    assert "import torch" not in policy_source

    orchestrator = next(
        node
        for node in ast.parse(exporter_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_fast_precanonicalize_repairs"
    )
    stored_names = {
        node.id
        for node in ast.walk(orchestrator)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert not stored_names.intersection(
        {
            "const_channel_counts",
            "conv_block_out_channels",
            "module_output_producers",
            "registered_buffer_shapes",
        }
    )
    orchestrator_source = ast.get_source_segment(exporter_source, orchestrator)
    assert orchestrator_source is not None
    assert orchestrator_source.index("_repair_binary_alignment_layout(") < (
        orchestrator_source.index(
            "_repair_binary_alignment_from_downstream_evidence("
        )
    )


def test_generated_pytorch_reshape_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_reshape_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_reshape_codegen_is_plain_data_only_for_codegen",
        "_reshape_prefers_feature_last_for_adjx_batch_matmul_for_codegen",
        "_static_sequence_length_for_model_ir",
        "_tensor_exact_static_shape_list_for_model_ir",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_nms_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_nms_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_is_identity_nms_postprocess_gather_for_codegen",
        "_range_only_feeds_identity_nms_postprocess_gathers_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_fusion_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_fusion_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_match_affine_layer_norm_for_codegen",
        "_match_if_axis0_tensor_mux_slice_for_codegen",
        "_match_swish_activation_pattern_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_constant_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_constant_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_axis_expr_from_input_for_codegen",
        "_constant_pad_pairs_for_codegen",
        "_int_scalar_literal_expr_for_codegen",
        "_is_constant_tensor_name_for_codegen",
        "_pad_literal_expr_for_codegen",
        "_reshape_shape_tensor_uses_runtime_dims_for_codegen",
        "_scalar_literal_expr_for_codegen",
        "_shape_tensor_constant_is_non_zero_int_vector_for_codegen",
        "_static_mirror_pad_expr_for_codegen",
        "_static_int_tensor_values_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_constant_alias_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_constant_alias_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_binary_trailing_axis_constant_buffer_alias_shape_for_codegen",
        "_channel_first_rank4_constant_buffer_alias_shape_for_codegen",
        "_constant_permute_for_broadcast_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_expression_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_expression_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_channel_first_constant_expr_for_buffer_attr_for_codegen",
        "_derived_local_var_name_for_codegen",
        "_permuted_constant_expr_for_tensor_name_for_codegen",
        "_tensor_dtype_name_for_codegen",
        "_tensor_expr_for_channel_first_bridge_for_codegen",
        "_tensor_expr_for_codegen",
        "_transposed_constant_expr_for_tensor_name_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_reduction_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_reduction_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_channel_first_reduction_plan_for_codegen",
        "_direct_mean_reduction_expr_for_codegen",
        "_normalized_constant_reduction_axes_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in policy_source


def test_generated_pytorch_shape_policy_has_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_shape_policy.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    policy_functions = {
        node.name
        for node in ast.parse(policy_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_conv2d_input_pre_permute_for_codegen",
        "_conv2d_output_spatial_shape_for_codegen",
        "_conv2d_same_pad_padding_arg_for_codegen",
        "_conv3d_output_spatial_shape_for_codegen",
        "_conv3d_transpose_output_spatial_shape_for_codegen",
        "_fast_precanonicalize_rank4_layout_hint",
        "_infer_batch_matmul_shape_for_codegen",
        "_infer_conv2d_ctor_params_for_codegen",
        "_infer_conv2d_layout_candidate_for_codegen",
        "_infer_conv3d_ctor_params_for_codegen",
        "_infer_conv3d_transpose_ctor_params_for_codegen",
        "_infer_reduction_shape_for_codegen",
        "_matmul_broadcast_shape_for_codegen",
        "_normalize_cf_rank4_shape",
        "_normalize_nhwc_rank4_shape",
        "_reshape_special_layout_plan",
        "_reshape_preserves_channel_last_sequence_for_codegen",
        "_should_skip_align_for_shape_preserving_unary_for_codegen",
        "_topk_codegen_layout_bridge_for_codegen",
    ):
        assert function_name in policy_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "_conv_output_spatial_shape" in policy_functions
    assert "_conv2d_same_pad_arg_for_codegen" not in exporter_functions
    assert "_conv2d_same_pad_arg_for_codegen" not in policy_functions
    assert "import torch" not in policy_source


def test_generated_pytorch_gap_se_rewrites_have_single_owner() -> None:
    exporter_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "pytorch_exporter.py"
    ).read_text(encoding="utf-8")
    rewrite_source = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "pytorch_source_rewrites.py"
    ).read_text(encoding="utf-8")
    exporter_functions = {
        node.name
        for node in ast.parse(exporter_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    rewrite_functions = {
        node.name
        for node in ast.parse(rewrite_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in (
        "_collapse_redundant_torch_permute_chains",
        "_fold_boundary_transpose_pad_conv_bridges",
        "_fold_channel_first_gap_conv_bridges",
        "_fold_channel_first_hardsigmoid_gate_conv_bridges",
        "_fold_channel_last_affine_conv_bridges",
        "_fold_channel_last_prelu_bridges",
        "_fold_rank4_reshape_permute_conv_bridges",
        "_inline_trivial_public_layout_bridge_aliases",
        "_prune_dead_forward_lines",
        "_repair_channel_last_gap_conv_inputs",
        "_repair_exported_program_direct_conv_cf_add_targets",
        "_rewrite_channel_first_gap_outputs_to_explicit_channel_last",
        "_rewrite_channel_first_se_scale_binary_bridges",
        "_rewrite_channel_last_binary_bridge_chains",
        "_rewrite_channel_last_gap_means_to_reduce_mean",
    ):
        assert function_name in rewrite_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
    assert "import torch" not in rewrite_source


def test_indexed_split_layout_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "split_channelwise_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plans_equal(" in owner_source
    assert "def _resolve_direct_candidate(" in owner_source
    assert "def _apply_direct_plan(" in owner_source
    assert "def _direct_plans_equal(" in owner_source
    assert "def _resolve_unary_split_concat_candidate(" in owner_source
    assert "def _apply_unary_split_concat_plan(" in owner_source
    assert "def _unary_split_concat_plans_equal(" in owner_source
    assert "def _resolve_closed_tail(" in owner_source
    assert "def _finalize_slice_constant_updates(" in owner_source
    assert "deque" in owner_source
    assert "work_limit" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "graph_index.remove_operator(" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    wrappers = {
        "_optimize_transpose_unary_split_concat_single_post_nchw": (
            "_optimize_transpose_unary_split_concat_single_post_nchw_pass"
        ),
        "_optimize_transpose_split_channelwise_tail_to_single_post_nchw": (
            "_optimize_transpose_split_channelwise_tail_to_single_post_nchw_pass"
        ),
        "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw": (
            "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw_pass"
        ),
    }
    for wrapper_name, dispatch_name in wrappers.items():
        wrapper = next(
            node
            for node in lowerer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        assert len(wrapper.body) == 1
        dispatch = wrapper.body[0]
        assert isinstance(dispatch, ast.Return)
        call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
        assert isinstance(call.func, ast.Name)
        assert call.func.id == dispatch_name
        assert {keyword.arg for keyword in call.keywords} == {
            "graph_index",
            "layout_state",
            "max_rewrites",
            "candidate",
        }

        production_calls = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == wrapper_name
        ]
        assert len(production_calls) == 2
        for production_call in production_calls:
            layout_keyword = next(
                keyword
                for keyword in production_call.keywords
                if keyword.arg == "layout_state"
            )
            assert isinstance(layout_keyword.value, ast.Attribute)
            assert isinstance(layout_keyword.value.value, ast.Name)
            assert layout_keyword.value.value.id == "session"
            assert layout_keyword.value.attr == "layout_state"


def test_indexed_split_adapter_owners_are_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "split_all_outputs_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def _resolve_conv_concat_candidate(" in owner_source
    assert "def _apply_conv_concat_plan(" in owner_source
    assert "def _conv_concat_plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrappers = {
        "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains": (
            "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains_pass"
        ),
        "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains": (
            "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains_pass"
        ),
    }
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    for wrapper_name, dispatch_name in wrappers.items():
        wrapper = next(
            node
            for node in lowerer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        assert len(wrapper.body) == 1
        dispatch = wrapper.body[0]
        assert isinstance(dispatch, ast.Return)
        call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
        assert isinstance(call.func, ast.Name)
        assert call.func.id == dispatch_name
        assert {keyword.arg for keyword in call.keywords} == {
            "graph_index",
            "layout_state",
            "max_rewrites",
            "candidate",
        }

        production_calls = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == wrapper_name
        ]
        assert len(production_calls) == 2
        for production_call in production_calls:
            layout_keyword = next(
                keyword
                for keyword in production_call.keywords
                if keyword.arg == "layout_state"
            )
            assert isinstance(layout_keyword.value, ast.Attribute)
            assert isinstance(layout_keyword.value.value, ast.Name)
            assert layout_keyword.value.value.id == "session"
            assert layout_keyword.value.attr == "layout_state"


def test_indexed_split_conv_concat_bridge_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "split_conv_concat_bridge_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def _reachable_before_concat(" in owner_source
    assert "edge_limit" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )
    dispatch_name = (
        "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw_pass"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_activation_passthrough_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "activation_passthrough_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def _resolve_gelu_candidate(" in owner_source
    assert "def _apply_gelu_plan(" in owner_source
    assert "def _gelu_plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("mobilevit", "yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    wrappers = {
        "_optimize_swish_transpose_passthrough_chains": (
            "_optimize_swish_transpose_passthrough_chains_pass",
            2,
        ),
        "_optimize_gelu_tanh_transpose_passthrough_chains": (
            "_optimize_gelu_tanh_transpose_passthrough_chains_pass",
            1,
        ),
    }
    for wrapper_name, (dispatch_name, expected_calls) in wrappers.items():
        wrapper = next(
            node
            for node in lowerer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        assert len(wrapper.body) == 1
        dispatch = wrapper.body[0]
        assert isinstance(dispatch, ast.Return)
        call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
        assert isinstance(call.func, ast.Name)
        assert call.func.id == dispatch_name
        assert {keyword.arg for keyword in call.keywords} == {
            "graph_index",
            "layout_state",
            "max_rewrites",
            "candidate",
        }
        production_calls = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == wrapper_name
        ]
        assert len(production_calls) == expected_calls
        for production_call in production_calls:
            layout_keyword = next(
                keyword
                for keyword in production_call.keywords
                if keyword.arg == "layout_state"
            )
            assert isinstance(layout_keyword.value, ast.Attribute)
            assert isinstance(layout_keyword.value.value, ast.Name)
            assert layout_keyword.value.value.id == "session"
            assert layout_keyword.value.attr == "layout_state"


def test_indexed_center_size_offset_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "center_size_offset_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def _constant_updates(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = "_optimize_center_size_offset_terminal_transpose_chains"
    dispatch_name = (
        "_optimize_center_size_offset_terminal_transpose_chains_pass"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_leakyrelu_passthrough_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "leakyrelu_passthrough_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def optimize_leakyrelu_transpose_passthrough(" in owner_source
    assert "def optimize_leakyrelu_transpose_passthrough_chains(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "graph_index=active_index" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = "_optimize_leakyrelu_transpose_passthrough_chains"
    dispatch_name = "_optimize_leakyrelu_transpose_passthrough_chains_pass"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_prelu_passthrough_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "prelu_passthrough_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def _selected_alpha(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = "_optimize_prelu_transpose_passthrough_chains"
    dispatch_name = "_optimize_prelu_transpose_passthrough_chains_pass"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_elementwise_concat_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "elementwise_concat_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def _resolve_boundary(" in owner_source
    assert "edge_limit" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = "_optimize_transpose_elementwise_concat_conv_nhwc_groups"
    dispatch_name = (
        "_optimize_transpose_elementwise_concat_conv_nhwc_groups_pass"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_singleton_gate_layout_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "singleton_gate_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operator(" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source

    wrapper_name = "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks"
    dispatch_name = (
        "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks_pass"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_stridedslice_concat_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "stridedslice_concat_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = "_optimize_transpose_stridedslice_pre_concat_nhwc_chains"
    dispatch_name = (
        "_optimize_transpose_stridedslice_pre_concat_nhwc_chains_pass"
    )
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_split_mixed_concat_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "split_mixed_concat_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains"
    )
    dispatch_name = f"{wrapper_name}_pass"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_concat_input_adapter_owner_preserves_all_call_boundaries() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "concat_input_adapter_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _resolve_branch(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yolov9", "sgscsh", "yunet", "humanseg", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter"
    )
    dispatch_name = f"{wrapper_name}_pass"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_pre_add_direct_unary_owner_precedes_compatibility_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_add_direct_unary_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compatibility_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_add_layout.py"
    )
    compatibility_source = compatibility_path.read_text(encoding="utf-8")
    compatibility_tree = ast.parse(compatibility_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)

    assert "def _resolve_candidate(" in owner_source
    assert "def _resolve_branch(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = "_optimize_transpose_pre_add_nhwc_chains"
    compatibility_name = "optimize_transpose_pre_add_nhwc_chains"
    compatibility_dispatch_name = (
        "_optimize_transpose_pre_add_nhwc_chains_pass"
    )
    indexed_dispatch_name = (
        "_optimize_transpose_pre_add_direct_unary_nhwc_chains_pass"
    )
    assert "lower_from_onnx2tf" not in compatibility_source
    compatibility_owner = next(
        node
        for node in compatibility_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compatibility_name
    )
    first_call = next(
        node
        for node in ast.walk(compatibility_owner.body[1])
        if isinstance(node, ast.Call)
    )
    assert isinstance(first_call.func, ast.Name)
    assert first_call.func.id == indexed_dispatch_name
    assert {keyword.arg for keyword in first_call.keywords} == {"layout_state"}
    assert "_build_tensor_consumer_map" in compatibility_source
    assert "_build_tensor_producer_map" in compatibility_source
    assert "_prune_unused_tensors" in compatibility_source
    assert "while True" in compatibility_source

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == compatibility_dispatch_name
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "layout_state"
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 4
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"

    safe_bundle = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_safe_transpose_reduction_lite"
    )
    pass_sequence = next(
        node
        for node in safe_bundle.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pass_sequence"
            for target in node.targets
        )
    )
    assert isinstance(pass_sequence.value, ast.List)
    assert [
        element.id
        for element in pass_sequence.value.elts
        if isinstance(element, ast.Name) and element.id == wrapper_name
    ] == [wrapper_name]


def test_dual_pre_add_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "dual_pre_add_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)

    owner_name = (
        "optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains"
    )
    wrapper_name = f"_{owner_name}"
    dispatch_name = f"{wrapper_name}_pass"
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert sum(
        isinstance(node, ast.FunctionDef) and node.name == owner_name
        for node in owner_tree.body
    ) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == dispatch_name

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1


def test_terminal_affine_fc_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_affine_fc_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    owner_name = "optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains"
    wrapper_name = f"_{owner_name}"
    dispatch_name = f"{wrapper_name}_pass"
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert sum(
        isinstance(node, ast.FunctionDef) and node.name == owner_name
        for node in owner_tree.body
    ) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == dispatch_name

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1


def test_terminal_prelu_bmm_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_prelu_bmm_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    owner_name = (
        "optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains"
    )
    wrapper_name = f"_{owner_name}"
    dispatch_name = f"{wrapper_name}_pass"
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert sum(
        isinstance(node, ast.FunctionDef) and node.name == owner_name
        for node in owner_tree.body
    ) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == dispatch_name

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1


def test_residual_affine_prelu_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "residual_affine_prelu_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    owner_name = "optimize_transpose_pre_add_mul_add_prelu_nhwc_chains"
    wrapper_name = f"_{owner_name}"
    dispatch_name = f"{wrapper_name}_pass"
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert sum(
        isinstance(node, ast.FunctionDef) and node.name == owner_name
        for node in owner_tree.body
    ) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [
        node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
    ]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == dispatch_name

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3


def test_residual_affine_fanout_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "residual_affine_fanout_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    owner_name = "optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains"
    wrapper_name = f"_{owner_name}"
    dispatch_name = f"{wrapper_name}_pass"
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "shared_outside_chain" in owner_source
    assert "legacy_users" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert sum(
        isinstance(node, ast.FunctionDef) and node.name == owner_name
        for node in owner_tree.body
    ) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == dispatch_name

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3


def test_pre_unary_affine_fanout_optimizer_has_one_module_owner() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_unary_affine_fanout_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    owner_name = "optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains"
    wrapper_name = f"_{owner_name}"
    dispatch_name = f"{wrapper_name}_pass"
    assert "lower_from_onnx2tf" not in owner_source
    assert "_build_tensor_consumer_map" in owner_source
    assert "_build_tensor_producer_map" in owner_source
    assert "shared_outside_chain" in owner_source
    assert "unary_ops" in owner_source
    assert "_prune_unused_tensors" in owner_source
    assert "while True" in owner_source
    assert sum(
        isinstance(node, ast.FunctionDef) and node.name == owner_name
        for node in owner_tree.body
    ) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == dispatch_name

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3


def test_indexed_pre_add_mulconst_reshape_suffix_owner_precedes_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_add_mulconst_reshape_suffix_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compat_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_add_mulconst_reshape_suffix_compat_layout.py"
    )
    compat_source = compat_path.read_text(encoding="utf-8")
    compat_tree = ast.parse(compat_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    removed_redundant_owner = (
        "_optimize_transpose_pre_add_reshape_transpose_suffix_nhwc_chains"
    )
    assert all(
        not (
            isinstance(node, ast.FunctionDef)
            and node.name == removed_redundant_owner
        )
        for node in ast.walk(lowerer_tree)
    )
    assert all(
        not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == removed_redundant_owner
        )
        for node in ast.walk(lowerer_tree)
    )

    assert "def _resolve_candidate(" in owner_source
    assert "def _resolve_branch(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("iat", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains"
    )
    dispatch_name = f"{wrapper_name}_pass"
    compat_owner_name = (
        "optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_compat"
    )
    compat_owner = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compat_owner_name
    )
    assert "lower_from_onnx2tf" not in compat_source
    assert "_build_tensor_consumer_map" in compat_source
    assert "_build_tensor_producer_map" in compat_source
    assert "_prune_unused_tensors" in compat_source
    assert "while True" in compat_source
    dispatch_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == dispatch_name
    ]
    assert len(dispatch_calls) == 1
    assert {keyword.arg for keyword in dispatch_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_compat_pass"
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "layout_state",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_pre_swish_reshape_suffix_owner_precedes_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_unary_reshape_suffix_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compat_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_unary_reshape_suffix_compat_layout.py"
    )
    compat_source = compat_path.read_text(encoding="utf-8")
    compat_tree = ast.parse(compat_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("linea", "iat", "yunet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains"
    )
    dispatch_name = (
        "_optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains_pass"
    )
    compat_owner_name = (
        "optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains_compat"
    )
    compat_owner = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compat_owner_name
    )
    assert "lower_from_onnx2tf" not in compat_source
    assert "_build_tensor_consumer_map" in compat_source
    assert "_build_tensor_producer_map" in compat_source
    assert "_prune_unused_tensors" in compat_source
    assert "while True" in compat_source
    dispatch_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == dispatch_name
    ]
    assert len(dispatch_calls) == 1
    assert {keyword.arg for keyword in dispatch_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }
    prune_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prune_unused_tensors"
    ]
    assert len(prune_calls) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_compat_pass"
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "layout_state",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_pre_swish_squeeze_suffix_owner_precedes_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_unary_squeeze_suffix_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compat_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "pre_unary_squeeze_suffix_compat_layout.py"
    )
    compat_source = compat_path.read_text(encoding="utf-8")
    compat_tree = ast.parse(compat_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("linea", "iat", "yunet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains"
    )
    dispatch_name = (
        "_optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains_pass"
    )
    compat_owner_name = (
        "optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat"
    )
    compat_owner = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compat_owner_name
    )
    assert "lower_from_onnx2tf" not in compat_source
    assert "_build_tensor_consumer_map" in compat_source
    assert "_build_tensor_producer_map" in compat_source
    assert "_prune_unused_tensors" in compat_source
    assert "while True" in compat_source
    dispatch_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == dispatch_name
    ]
    assert len(dispatch_calls) == 1
    assert {keyword.arg for keyword in dispatch_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }
    prune_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prune_unused_tensors"
    ]
    assert len(prune_calls) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_compat_pass"
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "layout_state",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 1
    layout_keyword = next(
        keyword
        for keyword in production_calls[0].keywords
        if keyword.arg == "layout_state"
    )
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"


def test_indexed_conv_mul_affine_owner_precedes_compat_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv_mul_affine_fold.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compat_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv_mul_affine_fold_compat.py"
    )
    compat_source = compat_path.read_text(encoding="utf-8")
    compat_tree = ast.parse(compat_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_source = lowerer_path.read_text(encoding="utf-8")
    lowerer_tree = ast.parse(lowerer_source)

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operator(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("iat", "linea", "yunet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    compat_owner_name = "optimize_fold_conv_mul_add_affine_chains"
    compat_owner = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compat_owner_name
    )
    dispatch_name = "_optimize_conv_mul_affine_mul_only_chains_pass"
    dispatch_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == dispatch_name
    ]
    assert len(dispatch_calls) == 1
    assert {keyword.arg for keyword in dispatch_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }
    prune_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prune_unused_tensors"
    ]
    assert len(prune_calls) == 1
    compat_owner_source = ast.get_source_segment(compat_source, compat_owner)
    assert compat_owner_source is not None
    assert "_build_tensor_consumer_map(model_ir)" in compat_owner_source
    assert "while True:" in compat_owner_source
    assert "del model_ir.operators[int(remove_idx)]" in compat_owner_source
    assert "lower_from_onnx2tf" not in compat_source

    wrapper_name = "_optimize_fold_conv_mul_add_affine_chains"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_dispatches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_optimize_fold_conv_mul_add_affine_chains_pass"
    ]
    assert len(wrapper_dispatches) == 1
    assert {keyword.arg for keyword in wrapper_dispatches[0].keywords} == {
        "enable_conv_add_only_fold",
        "layout_state",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 3
    for call in production_calls:
        layout_keyword = next(
            keyword for keyword in call.keywords if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_factorized_expanddims_owner_precedes_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "expanddims_reshape_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compat_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "expanddims_reshape_compat_layout.py"
    )
    compat_source = compat_path.read_text(encoding="utf-8")
    compat_tree = ast.parse(compat_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operator(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yolo", "nanodet", "linea", "iat", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains"
    )
    dispatch_name = "_optimize_transpose_factorized_expanddims_nhwc_chains_pass"
    compat_owner_name = (
        "optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat"
    )
    compat_owner = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compat_owner_name
    )
    assert "lower_from_onnx2tf" not in compat_source
    assert "_build_tensor_consumer_map" in compat_source
    assert "_prune_unused_tensors" in compat_source
    assert "while True" in compat_source
    dispatch_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == dispatch_name
    ]
    assert len(dispatch_calls) == 1
    assert {keyword.arg for keyword in dispatch_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }
    prune_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prune_unused_tensors"
    ]
    assert len(prune_calls) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_compat_pass"
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "layout_state",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_flatten_hw_reshape_owner_precedes_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "flatten_hw_reshape_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compat_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "flatten_hw_reshape_compat_layout.py"
    )
    compat_source = compat_path.read_text(encoding="utf-8")
    compat_tree = ast.parse(compat_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("linea", "yolo", "iat", "yunet", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains"
    )
    dispatch_name = "_optimize_transpose_flatten_hw_reshape_nhwc_chains_pass"
    compat_owner_name = (
        "optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat"
    )
    compat_owner = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compat_owner_name
    )
    assert "lower_from_onnx2tf" not in compat_source
    assert "_build_tensor_consumer_map" in compat_source
    assert "_prune_unused_tensors" in compat_source
    assert "while True" in compat_source
    dispatch_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == dispatch_name
    ]
    assert len(dispatch_calls) == 1
    assert {keyword.arg for keyword in dispatch_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }
    prune_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prune_unused_tensors"
    ]
    assert len(prune_calls) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_compat_pass"
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "layout_state",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_attention_qkv_reshape_owner_precedes_fallback() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "attention_qkv_reshape_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    compat_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "attention_qkv_reshape_compat_layout.py"
    )
    compat_source = compat_path.read_text(encoding="utf-8")
    compat_tree = ast.parse(compat_source)
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operator(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "_prune_unused_tensors" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("rfdetr", "htdemucs", "linea", "yolo", "iat", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains"
    )
    dispatch_name = "_optimize_attention_qkv_had_reshape_transpose_chains_pass"
    compat_owner_name = (
        "optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat"
    )
    compat_owner = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == compat_owner_name
    )
    assert "lower_from_onnx2tf" not in compat_source
    assert "_build_tensor_consumer_map" in compat_source
    assert "_prune_unused_tensors" in compat_source
    assert "while True" in compat_source
    dispatch_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == dispatch_name
    ]
    assert len(dispatch_calls) == 1
    assert {keyword.arg for keyword in dispatch_calls[0].keywords} == {
        "graph_index",
        "layout_state",
    }
    prune_calls = [
        node
        for node in ast.walk(compat_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prune_unused_tensors"
    ]
    assert len(prune_calls) == 1

    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    wrapper_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
    assert len(wrapper_calls) == 1
    assert isinstance(wrapper_calls[0].func, ast.Name)
    assert wrapper_calls[0].func.id == f"{wrapper_name}_compat_pass"
    assert {keyword.arg for keyword in wrapper_calls[0].keywords} == {
        "layout_state",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_slice_logistic_concat_reshape_tail_owner_is_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "slice_logistic_concat_reshape_tail_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _resolve_branch(" in owner_source
    assert "def _resolve_slice_path(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.insert_operator(" in owner_source
    assert "operator_indices_for_normalized_types(" in owner_source
    assert "max_rewrites" in owner_source
    assert "candidate" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("nanodet", "yolov9", "yunet", "humanseg", "sinet"):
        assert model_name not in owner_source.lower()

    wrapper_name = (
        "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"
    )
    dispatch_name = f"{wrapper_name}_pass"
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
    assert isinstance(call.func, ast.Name)
    assert call.func.id == dispatch_name
    assert {keyword.arg for keyword in call.keywords} == {
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    }

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper_name
    ]
    assert len(production_calls) == 2
    for production_call in production_calls:
        layout_keyword = next(
            keyword
            for keyword in production_call.keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"


def test_indexed_conv_output_passthrough_owner_is_bounded_and_transactional() -> None:
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "conv_output_passthrough_layout.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))

    assert "def _resolve_candidate(" in owner_source
    assert "def _apply_plan(" in owner_source
    assert "def _plan_signature(" in owner_source
    assert "def _resolve_terminal_candidate(" in owner_source
    assert "def _apply_terminal_plan(" in owner_source
    assert "def _terminal_plan_signature(" in owner_source
    assert "def _plan_rank4_constant_updates(" in owner_source
    assert "graph_index.remove_operators(" in owner_source
    assert "graph_index.remove_operator(" in owner_source
    assert "max_rewrites" in owner_source
    assert "_build_tensor_consumer_map" not in owner_source
    assert "_build_tensor_producer_map" not in owner_source
    assert "while True" not in owner_source
    for model_name in ("yunet", "fastestdet", "humanseg", "osnet", "sinet"):
        assert model_name not in owner_source.lower()

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    wrappers = {
        "_optimize_transposeconv_output_nhwc_passthrough_chains": (
            "_optimize_transposeconv_output_nhwc_passthrough_chains_pass"
        ),
        "_optimize_transposeconv_output_channel1_terminal_transpose_chains": (
            "_optimize_transposeconv_output_channel1_terminal_transpose_chains_pass"
        ),
    }
    for wrapper_name, dispatch_name in wrappers.items():
        wrapper = next(
            node
            for node in lowerer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
        )
        assert len(wrapper.body) == 1
        dispatch = wrapper.body[0]
        assert isinstance(dispatch, ast.Return)
        call = next(node for node in ast.walk(dispatch) if isinstance(node, ast.Call))
        assert isinstance(call.func, ast.Name)
        assert call.func.id == dispatch_name
        assert {keyword.arg for keyword in call.keywords} == {
            "graph_index",
            "layout_state",
            "max_rewrites",
            "candidate",
        }

        production_calls = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == wrapper_name
        ]
        assert len(production_calls) == 1
        layout_keyword = next(
            keyword
            for keyword in production_calls[0].keywords
            if keyword.arg == "layout_state"
        )
        assert isinstance(layout_keyword.value, ast.Attribute)
        assert isinstance(layout_keyword.value.value, ast.Name)
        assert layout_keyword.value.value.id == "session"
        assert layout_keyword.value.attr == "layout_state"
