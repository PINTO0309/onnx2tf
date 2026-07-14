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
    assert "ModelIRGraphIndex(model_ir)" in orphan_repair_source
    assert "graph_index.replace_operator_inputs(" in orphan_repair_source
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
    source = lowering_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_replace_expand_dims_and_squeeze_with_reshape"
    )
    function_source = ast.get_source_segment(source, function_node)
    assert function_source is not None
    assert "model_ir.operators =" not in function_source
    assert "ModelIRGraphIndex(model_ir)" in function_source
    assert "graph_index.insert_operator(" in function_source


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
    assert len(runner_calls) == 6


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
    assert len(runner_calls) == 5


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
    assert len(runner_calls) == 6


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
    assert len(runner_calls) == 7


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
    assert len(runner_calls) == 5


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
    assert len(calls) == 255
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
    assert len(hard_activation_calls) == 5
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
    assert len(reshape_only_duplicate_calls) == 4

    boundary_batchmatmul_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_boundary_input_batchmatmul_cleanup"
    ]
    assert len(boundary_batchmatmul_calls) == 4

    pad_mul_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_pad_mul_layout_cleanup"
    ]
    assert len(pad_mul_calls) == 3

    channel_slice_merge_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_channel_slice_merge_layout_cleanup"
    ]
    assert len(channel_slice_merge_calls) == 3

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
    assert len(quantized_prelu_calls) == 3

    quantized_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_quantized_reshape_cleanup"
    ]
    assert len(quantized_reshape_calls) == 3

    singleton_maxpool_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_maxpool_layout_cleanup"
    ]
    assert len(singleton_maxpool_calls) == 3

    singleton_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_reshape_layout_cleanup"
    ]
    assert len(singleton_reshape_calls) == 2

    consecutive_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_consecutive_reshape_cleanup"
    ]
    assert len(consecutive_reshape_calls) == 7

    flatten_concat_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_flatten_concat_reshape_cleanup"
    ]
    assert len(flatten_concat_reshape_calls) == 2

    singleton_spatial_reshape_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_spatial_reshape_cleanup"
    ]
    assert len(singleton_spatial_reshape_calls) == 2

    singleton_channel_transpose_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_singleton_channel_transpose_cleanup"
    ]
    assert len(singleton_channel_transpose_calls) == 5

    layout_transpose_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_layout_transpose_cleanup"
    ]
    assert len(layout_transpose_calls) == 13

    transpose_gather_axis_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_gather_axis_cleanup"
    ]
    assert len(transpose_gather_axis_calls) == 8

    transpose_gather_channel_fanout_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_gather_channel_fanout_cleanup"
    ]
    assert len(transpose_gather_channel_fanout_calls) == 4

    transpose_unary_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_unary_passthrough_cleanup"
    ]
    assert len(transpose_unary_calls) == 6

    transpose_unary_fanout_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_unary_fanout_bridge_cleanup"
    ]
    assert len(transpose_unary_fanout_calls) == 7

    transpose_unary_binary_fanout_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_transpose_unary_binary_fanout_bridge_cleanup"
    ]
    assert len(transpose_unary_binary_fanout_calls) == 6

    trailing_output_transpose_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_trailing_output_transpose_cleanup"
    ]
    assert len(trailing_output_transpose_calls) == 4

    nchw_channel_shuffle_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_nchw_channel_shuffle_cleanup"
    ]
    assert len(nchw_channel_shuffle_calls) == 6

    nhwc_channel_shuffle_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_nhwc_channel_shuffle_cleanup"
    ]
    assert len(nhwc_channel_shuffle_calls) == 5

    two_way_channel_shuffle_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_two_way_channel_shuffle_cleanup"
    ]
    assert len(two_way_channel_shuffle_calls) == 5

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
    assert len(transpose_mean_calls) == 6

    mean_mul_add_conv_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_mean_mul_add_conv_layout_cleanup"
    ]
    assert len(mean_mul_add_conv_calls) == 7

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
    assert len(terminal_mean_calls) == 6

    se_conv_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_se_conv_layout_cleanup"
    ]
    assert len(se_conv_calls) == 6

    se_fc_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_se_fc_layout_cleanup"
    ]
    assert len(se_fc_calls) == 9

    elementwise_gate_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_elementwise_gate_layout_cleanup"
    ]
    assert len(elementwise_gate_calls) == 5

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
    assert len(dual_postconv_gate_calls) == 5

    ndhwc_gate_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_ndhwc_gate_layout_cleanup"
    ]
    assert len(ndhwc_gate_calls) == 6

    cost_volume_scatter_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_cost_volume_scatter_layout_cleanup"
    ]
    assert len(cost_volume_scatter_calls) == 6

    add_concat_suffix_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_add_concat_suffix_layout_cleanup"
    ]
    assert len(add_concat_suffix_calls) == 5

    dual_mul_concat_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_dual_mul_concat_layout_cleanup"
    ]
    assert len(dual_mul_concat_calls) == 6


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
    assert "_constant_int_list," in exporter_source

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
    assert "def _build_named_encoder_methods(" not in exporter_source
    assert "import torch" not in stage_source


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
        "_parse_int_list_literal",
        "_strip_outer_parentheses",
        "_split_top_level_csv_exprs",
        "_parse_binary_add_args",
        "_parse_binary_mul_args",
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
        "_collapse_generated_name_tokens",
        "_direct_codegen_module_attr_base",
        "_extract_generated_name_suffix_tokens",
        "_make_tensor_storage_name_map",
        "_make_unique_identifier",
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
        "_direct_codegen_module_attr_base",
        "_make_tensor_storage_name_map",
        "_make_unique_identifier",
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

    moved_functions = (
        "_conv_block_activation_config",
        "_conv_block_activation_config_from_fused_name",
        "_is_small_inline_constant_tensor",
        "_python_literal_for_constant_tensor",
        "_scalar_literal_for_constant_tensor",
        "_torch_dtype_literal",
        "_torch_pad_literal_for_constant_tensor",
    )
    for function_name in moved_functions:
        assert function_name in values_functions
        assert function_name not in exporter_functions
        assert f"{function_name}," in exporter_source
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
    exporter_tree = ast.parse(exporter_source)
    exporter_functions = {
        node.name: node
        for node in exporter_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_build_model_ir_producer_consumer_index" not in exporter_functions
    writer = exporter_functions["_write_native_model_file"]
    constructor_calls = [
        node
        for node in ast.walk(writer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRGraphIndex"
    ]
    assert len(constructor_calls) == 1
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
        "_ensure_direct_codegen_supported",
        "_ensure_native_export_supported_ops",
        "_ensure_no_custom_ops",
        "_ensure_supported_ops",
        "_is_direct_codegen_unsupported_error",
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
