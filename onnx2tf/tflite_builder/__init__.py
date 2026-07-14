from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.artifact_metadata import (
    collect_custom_op_artifact_metadata,
)
from onnx2tf.tflite_builder.artifact_preparation import (
    isolate_float32_model_ir_for_tflite_write,
    resolve_requested_artifact_controls,
    resolve_requested_exporter_controls,
)
from onnx2tf.tflite_builder.core.contracts import (
    ArtifactPlan,
    ConversionRequest,
    ConversionResult,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.progress import (
    ProgressSpinner as _ProgressSpinner,
    create_progress_bar as _create_progress_bar,
)
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    summarize_model_ir_pass_diagnostics,
)
from onnx2tf.tflite_builder.core.validation import run_model_ir_validation_pipeline
from onnx2tf.tflite_builder.ir import (
    clone_model_ir_with_float16,
    clone_model_ir_with_float32,
    optimize_redundant_transpose_operators,
    prune_identity_cast_operators,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_constant_binary_elementwise_chains,
    _optimize_constant_input_scatter_nd_chains,
    build_op_coverage_report,
    build_tensor_correspondence_report,
    lower_onnx_to_ir,
    write_op_coverage_report,
    write_tensor_correspondence_report,
)
from onnx2tf.tflite_builder.model_writer import (
    MODEL_METADATA_ENTRIES_KEY,
    write_model_file,
)
from onnx2tf.tflite_builder.quantization import (
    StrictFullIntegerQuantizationError,
    build_dynamic_range_quantized_model_ir,
    build_full_integer_quantized_model_ir,
    build_full_integer_quantized_with_int16_act_model_ir,
    build_integer_quantized_model_ir,
    build_integer_quantized_with_int16_act_model_ir,
    collect_calibration_ranges_from_tflite,
    load_calibration_samples,
    strict_int16_activation_skip_reasons,
)

from onnx2tf.tflite_builder.preprocess import (
    configure_pseudo_ops_wave1_targets,
    get_supported_pseudo_ops_wave1_aliases,
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module
from onnx2tf.tflite_builder.split_planner import (
    crop_model_ir_by_boundary_tensors,
    DEFAULT_TFLITE_SPLIT_MAX_BYTES,
    DEFAULT_TFLITE_SPLIT_TARGET_BYTES,
    plan_contiguous_partitions_by_size,
    rewrite_model_ir_disable_group_convolution,
    rewrite_model_ir_unfold_batchmatmul,
    rewrite_model_ir_unroll_recurrent_ops,
    should_split_by_estimate,
    write_split_model_files_and_manifest,
    write_split_plan_report,
)
from onnx2tf.utils.onnx_litert_runtime import weights_export
from onnx2tf.utils.tf_optional import require_tensorflow
from onnx2tf.utils.torch_optional import require_torch

_INTERNAL_PASS_METRICS_PATH_ENV = "ONNX2TF_INTERNAL_PASS_METRICS_PATH"


def _write_internal_pass_metrics(
    path: str,
    diagnostics: List[Dict[str, Any]],
) -> None:
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = f"{output_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(
            summarize_model_ir_pass_diagnostics(diagnostics),
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    os.replace(tmp_path, output_path)


def _progress_write(*, message: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        from tqdm.auto import tqdm
        tqdm.write(str(message))
    except Exception:
        print(str(message), flush=True)


def _format_write_timing_line(
    *,
    stage: str,
    timing: Dict[str, Any],
) -> str:
    model_bytes = int(timing.get("model_bytes", 0))
    mb = float(model_bytes) / (1024.0 * 1024.0) if model_bytes > 0 else 0.0
    serializer_mode = str(timing.get("serializer_mode", "unknown"))
    build_sec = float(
        timing.get(
            "build_model_object_sec",
            timing.get("build_serialization_tables_sec", 0.0),
        )
    )
    return (
        "flatbuffer_direct write timing: "
        f"stage={stage} "
        f"mode={serializer_mode} "
        f"total={float(timing.get('write_model_file_total_sec', 0.0)):.3f}s "
        f"serialize={float(timing.get('serialize_total_sec', 0.0)):.3f}s "
        f"(sanitize={float(timing.get('sanitize_model_ir_sec', 0.0)):.3f}s "
        f"build={build_sec:.3f}s "
        f"pack={float(timing.get('pack_builder_sec', 0.0)):.3f}s "
        f"output={float(timing.get('output_buffer_sec', 0.0)):.3f}s) "
        f"write={float(timing.get('file_write_sec', 0.0)):.3f}s "
        f"size={mb:.2f}MB"
    )


def _build_export_progress_labels(
    *,
    artifact_plan: ArtifactPlan,
) -> List[str]:
    labels: List[str] = [
        "tensor correspondence report",
    ]
    if artifact_plan.op_coverage_report:
        labels.append("op coverage report")
    if artifact_plan.split_manifest:
        labels.append("split planning")
    if artifact_plan.float32_tflite:
        labels.append("write float32 tflite")
    if artifact_plan.saved_model:
        labels.append("write saved_model")
    if artifact_plan.pytorch:
        labels.append("write pytorch")
    if artifact_plan.float16_tflite:
        labels.append("write float16 tflite")
    if artifact_plan.dynamic_range_quantized_tflite:
        labels.append("write dynamic range quant tflite")
    if artifact_plan.integer_quantized_tflite:
        labels.extend(
            [
                "write integer quant tflite",
                "write full integer quant tflite",
                "write integer quant int16-act tflite",
                "write full integer quant int16-act tflite",
            ]
        )
    if artifact_plan.weights:
        labels.extend(
            [
                "export float32 weights",
                "export float16 weights",
            ]
        )
        if artifact_plan.dynamic_range_quantized_tflite:
            labels.append("export dynamic range quant weights")
        if artifact_plan.integer_quantized_tflite:
            labels.extend(
                [
                    "export integer quant weights",
                    "export full integer quant weights",
                    "export integer quant int16-act weights",
                    "export full integer quant int16-act weights",
                ]
            )
    return labels


def _reject_unsupported_quantization(**kwargs: Any) -> None:
    return


def _set_reduced_precision_support_metadata(
    *,
    model_ir: Any,
    enable_accumulation_type_float16: bool,
) -> None:
    reduced_precision_hint = (
        b"fp16accfp16"
        if bool(enable_accumulation_type_float16)
        else b"fp16accfp32"
    )
    metadata = getattr(model_ir, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}

    existing_entries_raw = metadata.get(MODEL_METADATA_ENTRIES_KEY, [])
    if isinstance(existing_entries_raw, dict):
        existing_entries = [existing_entries_raw]
    elif isinstance(existing_entries_raw, (list, tuple)):
        existing_entries = list(existing_entries_raw)
    else:
        existing_entries = []

    preserved_entries: List[Dict[str, Any]] = []
    for entry in existing_entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name", "")).strip() == "reduced_precision_support":
            continue
        preserved_entries.append(dict(entry))

    preserved_entries.append(
        {
            "name": "reduced_precision_support",
            "data": reduced_precision_hint,
        }
    )
    metadata[MODEL_METADATA_ENTRIES_KEY] = preserved_entries
    model_ir.metadata = metadata


def export_tflite_model_flatbuffer_direct(**kwargs: Any) -> Dict[str, Any]:
    request = ConversionRequest.from_kwargs(kwargs)
    _reject_unsupported_quantization(**request.options)

    output_folder_path = request.output_folder_path
    output_file_name = request.output_file_name
    onnx_graph = request.onnx_graph
    output_weights = request.artifacts.weights
    output_dynamic_range_quantized_tflite = request.artifacts.dynamic_range_quantized_tflite
    output_integer_quantized_tflite = request.artifacts.integer_quantized_tflite
    output_saved_model_from_model_ir = request.artifacts.saved_model
    output_pytorch_from_model_ir = request.artifacts.pytorch
    output_torchscript_from_model_ir = request.artifacts.torchscript
    output_dynamo_onnx_from_model_ir = request.artifacts.dynamo_onnx
    output_exported_program_from_model_ir = request.artifacts.exported_program
    required_pytorch_feature: Optional[str] = None
    if output_torchscript_from_model_ir:
        required_pytorch_feature = "flatbuffer_direct TorchScript export"
    elif output_dynamo_onnx_from_model_ir:
        required_pytorch_feature = "flatbuffer_direct Dynamo ONNX export"
    elif output_exported_program_from_model_ir:
        required_pytorch_feature = "flatbuffer_direct ExportedProgram export"
    elif output_pytorch_from_model_ir:
        required_pytorch_feature = "flatbuffer_direct PyTorch package export"
    if required_pytorch_feature is not None:
        require_torch(required_pytorch_feature)
    exporter_controls = resolve_requested_exporter_controls(
        request.options,
        output_folder_path=output_folder_path,
        output_file_name=output_file_name,
        artifact_plan=request.artifacts,
    )
    saved_model_output_folder_path = (
        exporter_controls.saved_model_output_folder_path
    )
    pytorch_output_folder_path = exporter_controls.pytorch_output_folder_path
    persist_saved_model_output = exporter_controls.persist_saved_model_output
    enable_accumulation_type_float16 = bool(
        request.get("enable_accumulation_type_float16", False)
    )
    force_split_manifest = request.artifacts.split_manifest
    split_plan_requested = bool(force_split_manifest)
    report_op_coverage = request.artifacts.op_coverage_report
    custom_input_op_name_np_data_path = (
        exporter_controls.custom_input_op_name_np_data_path
    )
    shape_hints = exporter_controls.shape_hints
    test_data_nhwc_path = exporter_controls.test_data_nhwc_path
    native_pytorch_generation_timeout_sec = (
        exporter_controls.native_pytorch_generation_timeout_sec
    )
    output_nms_with_argmax = bool(request.get("output_nms_with_argmax", False))
    switch_nms_version = str(request.get("switch_nms_version", "v4")).strip().lower()
    if switch_nms_version not in {"v4", "v5"}:
        raise ValueError(
            "switch_nms_version must be 'v4' or 'v5'. "
            f"got: {switch_nms_version}"
        )
    keep_ncw_or_nchw_or_ncdhw_input_names = request.get(
        "keep_ncw_or_nchw_or_ncdhw_input_names",
        None,
    )
    keep_nwc_or_nhwc_or_ndhwc_input_names = request.get(
        "keep_nwc_or_nhwc_or_ndhwc_input_names",
        None,
    )
    keep_shape_absolutely_input_names = request.get(
        "keep_shape_absolutely_input_names",
        None,
    )
    disable_group_convolution = bool(
        request.get("disable_group_convolution", False)
    )
    enable_batchmatmul_unfold = bool(
        request.get("enable_batchmatmul_unfold", False)
    )
    enable_rnn_unroll = bool(
        request.get("enable_rnn_unroll", False)
    )
    mvn_epsilon = float(request.get("mvn_epsilon", 1e-10))
    flatbuffer_direct_allow_custom_ops = bool(
        request.get("flatbuffer_direct_allow_custom_ops", False)
    )
    custom_allowlist_raw = request.get(
        "flatbuffer_direct_custom_op_allowlist",
        None,
    )
    if custom_allowlist_raw is None:
        flatbuffer_direct_custom_op_allowlist = None
    elif isinstance(custom_allowlist_raw, (list, tuple, set)):
        flatbuffer_direct_custom_op_allowlist = [
            str(v).strip() for v in custom_allowlist_raw if str(v).strip() != ""
        ]
    else:
        v = str(custom_allowlist_raw).strip()
        flatbuffer_direct_custom_op_allowlist = [v] if v != "" else None
    artifact_execution_controls = resolve_requested_artifact_controls(
        request.options,
        artifact_plan=request.artifacts,
        default_split_max_bytes=DEFAULT_TFLITE_SPLIT_MAX_BYTES,
        default_split_target_bytes=DEFAULT_TFLITE_SPLIT_TARGET_BYTES,
    )
    tflite_split_max_bytes = artifact_execution_controls.split_max_bytes
    tflite_split_target_bytes = artifact_execution_controls.split_target_bytes
    quant_controls = artifact_execution_controls.quantization
    if split_plan_requested and (
        tflite_split_max_bytes is None or tflite_split_target_bytes is None
    ):
        raise RuntimeError("split artifact controls were not resolved")
    if (
        output_dynamic_range_quantized_tflite
        or output_integer_quantized_tflite
    ) and quant_controls is None:
        raise RuntimeError("quantization artifact controls were not resolved")
    quant_type = (
        quant_controls["quant_type"]
        if quant_controls is not None
        else "per-channel"
    )
    input_quant_dtype = (
        quant_controls["input_quant_dtype"]
        if quant_controls is not None
        else "int8"
    )
    output_quant_dtype = (
        quant_controls["output_quant_dtype"]
        if quant_controls is not None
        else "int8"
    )
    flatbuffer_direct_show_progress = bool(
        request.get("flatbuffer_direct_show_progress", True)
    )
    number_of_dimensions_after_flextranspose_compression = int(
        request.get("number_of_dimensions_after_flextranspose_compression", 6)
    )
    disable_suppression_flextranspose = bool(
        request.get("disable_suppression_flextranspose", False)
    )
    number_of_dimensions_after_flexstridedslice_compression = int(
        request.get("number_of_dimensions_after_flexstridedslice_compression", 5)
    )
    disable_suppression_flexstridedslice = bool(
        request.get("disable_suppression_flexstridedslice", False)
    )
    optimization_for_gpu_delegate = bool(
        request.get("optimization_for_gpu_delegate", False)
    )
    replace_argmax_to_reducemax_and_indices_is_int64 = bool(
        request.get("replace_argmax_to_reducemax_and_indices_is_int64", False)
    )
    replace_argmax_to_reducemax_and_indices_is_float32 = bool(
        request.get("replace_argmax_to_reducemax_and_indices_is_float32", False)
    )
    replace_argmax_to_fused_argmax_and_indices_is_int64 = bool(
        request.get("replace_argmax_to_fused_argmax_and_indices_is_int64", False)
    )
    replace_argmax_to_fused_argmax_and_indices_is_float32 = bool(
        request.get("replace_argmax_to_fused_argmax_and_indices_is_float32", False)
    )
    fused_argmax_scale_ratio = float(
        request.get("fused_argmax_scale_ratio", 0.5)
    )
    requested_pseudo_ops_raw = request.get("replace_to_pseudo_operators", None)
    input_names_to_interrupt_model_conversion = request.get(
        "input_names_to_interrupt_model_conversion",
        None,
    )
    output_names_to_interrupt_model_conversion = request.get(
        "output_names_to_interrupt_model_conversion",
        None,
    )
    if requested_pseudo_ops_raw is None:
        requested_pseudo_ops: List[str] = []
    elif isinstance(requested_pseudo_ops_raw, (list, tuple, set)):
        requested_pseudo_ops = [
            str(v).strip().lower()
            for v in requested_pseudo_ops_raw
            if str(v).strip() != ""
        ]
    else:
        requested_pseudo_ops = [
            str(requested_pseudo_ops_raw).strip().lower()
        ] if str(requested_pseudo_ops_raw).strip() != "" else []
    supported_pseudo_aliases = set(get_supported_pseudo_ops_wave1_aliases())
    ignored_requested_pseudo_ops = sorted(
        list(
            {
                str(alias) for alias in requested_pseudo_ops
                if str(alias) not in supported_pseudo_aliases
            }
        )
    )
    if len(ignored_requested_pseudo_ops) > 0:
        _progress_write(
            message=(
                "flatbuffer_direct: ignored unsupported --replace_to_pseudo_operators entries. "
                f"unsupported={ignored_requested_pseudo_ops} "
                f"supported={sorted(list(supported_pseudo_aliases))}"
            ),
            enabled=bool(flatbuffer_direct_show_progress),
        )

    if onnx_graph is None:
        raise ValueError(
            "onnx_graph is required for tflite_backend='flatbuffer_direct'."
        )
    register_default_preprocess_rules()
    preprocessed_onnx_graph = onnx_graph
    preprocess_report: Dict[str, Any] = {
        "schema_version": 1,
        "pipeline_version": 1,
        "registered_rule_ids": [],
        "enabled_rule_ids": [],
        "applied_rules": [],
        "summary": {
            "registered_rule_count": 0,
            "enabled_rule_count": 0,
            "executed_rule_count": 0,
            "changed_rule_count": 0,
            "total_matched_nodes": 0,
            "total_rewritten_nodes": 0,
        },
    }

    os.makedirs(output_folder_path, exist_ok=True)
    schema_tflite = load_schema_module(output_folder_path)

    op_coverage_report_path = None
    if report_op_coverage:
        op_coverage_report_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_op_coverage_report.json",
        )

    def _write_coverage_report(conversion_error: str | None) -> None:
        if not report_op_coverage or op_coverage_report_path is None:
            return
        report = build_op_coverage_report(
            onnx_graph=preprocessed_onnx_graph,
            output_file_name=output_file_name,
            conversion_error=conversion_error,
            allow_custom_ops=flatbuffer_direct_allow_custom_ops,
            custom_op_allowlist=flatbuffer_direct_custom_op_allowlist,
            disable_group_convolution=disable_group_convolution,
            preprocess_report=preprocess_report,
            output_nms_with_argmax=output_nms_with_argmax,
            switch_nms_version=switch_nms_version,
        )
        write_op_coverage_report(
            report=report,
            output_report_path=op_coverage_report_path,
        )

    configure_pseudo_ops_wave1_targets(
        requested_pseudo_ops if len(requested_pseudo_ops) > 0 else None
    )
    try:
        try:
            preprocessed_onnx_graph, preprocess_report = run_preprocess_pipeline(
                onnx_graph=onnx_graph,
            )
        except Exception as ex:
            if report_op_coverage:
                try:
                    _write_coverage_report(str(ex))
                except Exception:
                    pass
            raise
    finally:
        configure_pseudo_ops_wave1_targets(None)

    internal_pass_metrics_path = str(
        os.environ.get(_INTERNAL_PASS_METRICS_PATH_ENV, "")
    ).strip()
    internal_pass_diagnostics: Optional[List[Dict[str, Any]]] = (
        [] if internal_pass_metrics_path else None
    )
    split_graph_index: Optional[ModelIRGraphIndex] = None
    try:
        model_ir = lower_onnx_to_ir(
            onnx_graph=preprocessed_onnx_graph,
            output_file_name=output_file_name,
            allow_custom_ops=flatbuffer_direct_allow_custom_ops,
            custom_op_allowlist=flatbuffer_direct_custom_op_allowlist,
            transpose_inputs_to_nhwc=True,
            keep_ncw_or_nchw_or_ncdhw_input_names=keep_ncw_or_nchw_or_ncdhw_input_names,
            keep_nwc_or_nhwc_or_ndhwc_input_names=keep_nwc_or_nhwc_or_ndhwc_input_names,
            keep_shape_absolutely_input_names=keep_shape_absolutely_input_names,
            disable_group_convolution=disable_group_convolution,
            output_nms_with_argmax=output_nms_with_argmax,
            switch_nms_version=switch_nms_version,
            mvn_epsilon=mvn_epsilon,
            show_progress=flatbuffer_direct_show_progress,
            disable_suppression_flextranspose=disable_suppression_flextranspose,
            number_of_dimensions_after_flextranspose_compression=number_of_dimensions_after_flextranspose_compression,
            disable_suppression_flexstridedslice=disable_suppression_flexstridedslice,
            number_of_dimensions_after_flexstridedslice_compression=number_of_dimensions_after_flexstridedslice_compression,
            optimization_for_gpu_delegate=optimization_for_gpu_delegate,
            replace_argmax_to_reducemax_and_indices_is_int64=replace_argmax_to_reducemax_and_indices_is_int64,
            replace_argmax_to_reducemax_and_indices_is_float32=replace_argmax_to_reducemax_and_indices_is_float32,
            replace_argmax_to_fused_argmax_and_indices_is_int64=replace_argmax_to_fused_argmax_and_indices_is_int64,
            replace_argmax_to_fused_argmax_and_indices_is_float32=replace_argmax_to_fused_argmax_and_indices_is_float32,
            fused_argmax_scale_ratio=fused_argmax_scale_ratio,
            replace_to_pseudo_operators=requested_pseudo_ops,
            protected_boundary_tensor_names=list(
                dict.fromkeys(
                    list(input_names_to_interrupt_model_conversion or [])
                    + list(output_names_to_interrupt_model_conversion or [])
                )
            ),
            _internal_pass_diagnostics=internal_pass_diagnostics,
        )
        if internal_pass_diagnostics is not None:
            try:
                _write_internal_pass_metrics(
                    internal_pass_metrics_path,
                    internal_pass_diagnostics,
                )
            except Exception:
                pass
        if (
            input_names_to_interrupt_model_conversion
            or output_names_to_interrupt_model_conversion
        ):
            default_output_boundaries = (
                list(model_ir.metadata.get("original_graph_output_names", []))
                if (
                    output_names_to_interrupt_model_conversion is None
                    and input_names_to_interrupt_model_conversion
                )
                else None
            )
            model_ir = crop_model_ir_by_boundary_tensors(
                model_ir=model_ir,
                requested_inputs=input_names_to_interrupt_model_conversion,
                requested_outputs=(
                    output_names_to_interrupt_model_conversion
                    if output_names_to_interrupt_model_conversion is not None
                    else default_output_boundaries
                ),
            )
        if disable_group_convolution:
            model_ir, _ = rewrite_model_ir_disable_group_convolution(
                model_ir=model_ir,
            )
        if enable_batchmatmul_unfold:
            model_ir, _ = rewrite_model_ir_unfold_batchmatmul(
                model_ir=model_ir,
            )
        if enable_rnn_unroll:
            model_ir, _ = rewrite_model_ir_unroll_recurrent_ops(
                model_ir=model_ir,
            )
        if split_plan_requested:
            split_graph_index = ModelIRGraphIndex(model_ir)
        run_model_ir_validation_pipeline(
            model_ir,
            graph_index=split_graph_index,
        )
    except Exception as ex:
        if report_op_coverage:
            try:
                _write_coverage_report(str(ex))
            except Exception:
                pass
        raise

    export_progress_labels = _build_export_progress_labels(
        artifact_plan=request.artifacts,
    )
    export_progress_total = int(len(export_progress_labels))
    export_progress_step = 0
    export_progress_bar = _create_progress_bar(
        total=export_progress_total,
        desc="flatbuffer_direct export",
        enabled=bool(flatbuffer_direct_show_progress),
    )
    export_progress_spinner = _ProgressSpinner(export_progress_bar)

    def _set_export_progress_desc(stage_label: str) -> None:
        if export_progress_bar is None:
            return
        export_progress_spinner.start()
        export_progress_bar.set_description_str(
            f"flatbuffer_direct export [{export_progress_step + 1}/{export_progress_total}] {stage_label}"
        )

    def _advance_export_progress() -> None:
        nonlocal export_progress_step
        if export_progress_bar is None:
            return
        export_progress_spinner.stop()
        export_progress_bar.update(1)
        export_progress_step = int(export_progress_step + 1)

    tensor_correspondence_report_path = os.path.join(
        output_folder_path,
        f"{output_file_name}_tensor_correspondence_report.json",
    )
    try:
        _set_export_progress_desc("tensor correspondence report")
        tensor_correspondence_report = build_tensor_correspondence_report(
            onnx_graph=preprocessed_onnx_graph,
            model_ir=model_ir,
        )
        write_tensor_correspondence_report(
            report=tensor_correspondence_report,
            output_report_path=tensor_correspondence_report_path,
        )
        _advance_export_progress()

        if report_op_coverage:
            _set_export_progress_desc("op coverage report")
            _write_coverage_report(None)
            _advance_export_progress()

        custom_ops_used, custom_op_nodes = collect_custom_op_artifact_metadata(
            model_ir
        )

        split_plan_report_path = None
        split_required_by_estimate = False
        split_plan_total_estimated_bytes = None
        split_manifest_path = None
        split_partition_paths = None
        split_partition_count = 0
        write_timing_report: Dict[str, Dict[str, Any]] = {}
        split_saved_model_dirs = None
        pytorch_package_path = None
        pytorch_torchscript_path = None
        pytorch_dynamo_onnx_path = None
        pytorch_exported_program_path = None
        split_pytorch_package_dirs = None
        split_pytorch_torchscript_paths = None
        split_pytorch_dynamo_onnx_paths = None
        split_pytorch_exported_program_paths = None
        calibration_report_path = None
        if split_plan_requested:
            _set_export_progress_desc("split planning")
            split_plan_report = plan_contiguous_partitions_by_size(
                model_ir=model_ir,
                target_max_bytes=tflite_split_target_bytes,
                hard_max_bytes=tflite_split_max_bytes,
                schema_tflite=schema_tflite,
                graph_index=split_graph_index,
            )
            split_required_by_estimate = bool(should_split_by_estimate(split_plan_report))
            split_plan_total_estimated_bytes = int(
                split_plan_report.get("total_estimated_bytes", 0)
            )
            split_plan_report_path = write_split_plan_report(
                report=split_plan_report,
                output_report_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_split_plan.json",
                ),
            )
            if split_required_by_estimate or force_split_manifest:
                from ai_edge_litert.interpreter import Interpreter

                def _validate_split_tflite_loadable(tflite_path: str) -> None:
                    interpreter = Interpreter(model_path=tflite_path)
                    interpreter.allocate_tensors()

                split_outputs = write_split_model_files_and_manifest(
                    schema_tflite=schema_tflite,
                    model_ir=model_ir,
                    plan_report=split_plan_report,
                    output_folder_path=output_folder_path,
                    output_file_name=output_file_name,
                    tflite_loader_validator=_validate_split_tflite_loadable,
                    graph_index=split_graph_index,
                )
                split_manifest_path = split_outputs["split_manifest_path"]
                split_partition_paths = split_outputs["split_partition_paths"]
                split_partition_count = int(split_outputs["split_partition_count"])
            _advance_export_progress()
            del split_graph_index

        _set_export_progress_desc("write float32 tflite")
        model_ir_fp32 = clone_model_ir_with_float32(model_ir)
        prune_identity_cast_operators(
            model_ir_fp32,
            preserve_model_outputs=True,
        )
        optimize_redundant_transpose_operators(
            model_ir_fp32,
            preserve_model_outputs=True,
        )
        _set_reduced_precision_support_metadata(
            model_ir=model_ir_fp32,
            enable_accumulation_type_float16=enable_accumulation_type_float16,
        )
        float32_path = os.path.join(output_folder_path, f"{output_file_name}_float32.tflite")
        saved_model_path = None
        float32_write_timing: Dict[str, Any] = {}
        model_ir_fp32_tflite = isolate_float32_model_ir_for_tflite_write(
            model_ir_fp32,
            split_manifest_path=split_manifest_path,
            output_saved_model_from_model_ir=output_saved_model_from_model_ir,
            output_pytorch_from_model_ir=output_pytorch_from_model_ir,
        )
        fp32_write_graph_index = ModelIRGraphIndex(model_ir_fp32_tflite)
        _optimize_constant_input_scatter_nd_chains(
            model_ir_fp32_tflite,
            graph_index=fp32_write_graph_index,
        )
        _optimize_constant_binary_elementwise_chains(
            model_ir_fp32_tflite,
            graph_index=fp32_write_graph_index,
        )
        run_model_ir_validation_pipeline(
            model_ir_fp32_tflite,
            graph_index=fp32_write_graph_index,
        )
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=model_ir_fp32_tflite,
            output_tflite_path=float32_path,
            timing=float32_write_timing,
        )
        write_timing_report["float32"] = float32_write_timing
        _progress_write(
            message=_format_write_timing_line(
                stage="float32",
                timing=float32_write_timing,
            ),
            enabled=bool(flatbuffer_direct_show_progress),
        )
        _advance_export_progress()
        del fp32_write_graph_index
        del model_ir_fp32_tflite

        calibration_ranges = None
        calibration_report: Dict[str, Any] = {}
        if output_integer_quantized_tflite:
            calibration_samples = load_calibration_samples(
                custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                input_names=[str(v) for v in model_ir.inputs],
            )
            calibration_ranges = collect_calibration_ranges_from_tflite(
                tflite_path=float32_path,
                calibration_samples=calibration_samples,
            )
            calibration_report = {
                "sample_count": int(len(calibration_samples)),
                "tensor_ranges": {
                    str(name): {
                        "min": float(value.min_value),
                        "max": float(value.max_value),
                        "num_samples": int(value.num_samples),
                    }
                    for name, value in calibration_ranges.items()
                },
                "variants": {},
            }
            del calibration_samples

        def _write_validated_quantized_model_file(
            *,
            model_ir: Any,
            output_tflite_path: str,
            timing: Dict[str, Any],
        ) -> None:
            run_model_ir_validation_pipeline(model_ir)
            tmp_path = f"{output_tflite_path}.tmp"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            write_model_file(
                schema_tflite=schema_tflite,
                model_ir=model_ir,
                output_tflite_path=tmp_path,
                timing=timing,
            )
            try:
                from ai_edge_litert.interpreter import Interpreter, OpResolverType

                try:
                    interpreter = Interpreter(
                        model_path=tmp_path,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                    )
                except TypeError:
                    interpreter = Interpreter(model_path=tmp_path)
                interpreter.allocate_tensors()
            except Exception as ex:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise StrictFullIntegerQuantizationError(
                    f"Strict integer quantized TFLite validation failed: {output_tflite_path}: {ex}"
                ) from ex
            os.replace(tmp_path, output_tflite_path)

        if output_saved_model_from_model_ir:
            require_tensorflow("flatbuffer_direct SavedModel export")
            _set_export_progress_desc("write saved_model")
            if split_manifest_path is not None:
                from onnx2tf.tflite_builder.split_saved_model_exporter import (
                    export_split_saved_models,
                )

                split_saved_model_outputs = export_split_saved_models(
                    model_ir=model_ir,
                    split_manifest_path=split_manifest_path,
                    output_folder_path=output_folder_path,
                    output_file_name=output_file_name,
                )
                split_saved_model_dirs = split_saved_model_outputs["split_saved_model_dirs"]
            else:
                from onnx2tf.tflite_builder.saved_model_exporter import (
                    export_saved_model_from_model_ir,
                )
                saved_model_path = export_saved_model_from_model_ir(
                    model_ir=model_ir_fp32,
                    output_folder_path=saved_model_output_folder_path,
                )
            _advance_export_progress()

        if output_pytorch_from_model_ir:
            require_torch(required_pytorch_feature or "flatbuffer_direct PyTorch package export")
            _set_export_progress_desc("write pytorch")
            if split_manifest_path is not None:
                from onnx2tf.tflite_builder.split_pytorch_exporter import (
                    export_split_pytorch_packages,
                )

                split_pytorch_outputs = export_split_pytorch_packages(
                    model_ir=model_ir,
                    split_manifest_path=split_manifest_path,
                    output_folder_path=output_folder_path,
                    output_file_name=output_file_name,
                    output_torchscript_from_model_ir=output_torchscript_from_model_ir,
                    output_dynamo_onnx_from_model_ir=output_dynamo_onnx_from_model_ir,
                    output_exported_program_from_model_ir=output_exported_program_from_model_ir,
                    custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                    shape_hints=shape_hints,
                    test_data_nhwc_path=test_data_nhwc_path,
                    native_package_generation_timeout_sec=native_pytorch_generation_timeout_sec,
                )
                split_pytorch_package_dirs = split_pytorch_outputs["split_pytorch_package_dirs"]
                split_pytorch_torchscript_paths = split_pytorch_outputs.get(
                    "split_pytorch_torchscript_paths",
                    None,
                )
                split_pytorch_dynamo_onnx_paths = split_pytorch_outputs.get(
                    "split_pytorch_dynamo_onnx_paths",
                    None,
                )
                split_pytorch_exported_program_paths = split_pytorch_outputs.get(
                    "split_pytorch_exported_program_paths",
                    None,
                )
            else:
                from onnx2tf.tflite_builder.pytorch_exporter import (
                    NativePyTorchGenerationTimeoutError,
                    export_dynamo_onnx_from_generated_package,
                    export_exported_program_from_generated_package,
                    export_pytorch_package_from_model_ir,
                    export_torchscript_from_generated_package,
                )
                fallback_saved_model_path = saved_model_path

                def _ensure_pytorch_saved_model_bridge() -> Optional[str]:
                    nonlocal fallback_saved_model_path
                    if fallback_saved_model_path is not None and str(fallback_saved_model_path).strip() != "":
                        return str(fallback_saved_model_path)
                    try:
                        require_tensorflow("flatbuffer_direct PyTorch SavedModel bridge")
                        from onnx2tf.tflite_builder.saved_model_exporter import (
                            export_saved_model_from_model_ir,
                        )
                        fallback_saved_model_path = export_saved_model_from_model_ir(
                            model_ir=model_ir_fp32,
                            output_folder_path=os.path.join(
                                output_folder_path,
                                f".{output_file_name}_pytorch_saved_model_bridge",
                            ),
                        )
                    except Exception:
                        fallback_saved_model_path = None
                    return (
                        str(fallback_saved_model_path)
                        if fallback_saved_model_path is not None and str(fallback_saved_model_path).strip() != ""
                        else None
                    )

                try:
                    pytorch_package_path = export_pytorch_package_from_model_ir(
                        model_ir=model_ir_fp32,
                        output_folder_path=pytorch_output_folder_path,
                        fallback_tflite_path=float32_path,
                        fallback_onnx_graph=onnx_graph,
                        fallback_saved_model_path=fallback_saved_model_path,
                        fallback_saved_model_factory=_ensure_pytorch_saved_model_bridge,
                        fallback_tflite_has_custom_ops=bool(len(custom_ops_used) > 0),
                        native_package_generation_timeout_sec=native_pytorch_generation_timeout_sec,
                    )
                except NativePyTorchGenerationTimeoutError as ex:
                    pytorch_package_path = None
                    _progress_write(
                        message=f"PyTorch package generation skipped. {ex}",
                        enabled=bool(flatbuffer_direct_show_progress),
                    )
                if pytorch_package_path is not None and output_torchscript_from_model_ir:
                    pytorch_torchscript_path = export_torchscript_from_generated_package(
                        package_dir=str(pytorch_package_path),
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                        shape_hints=shape_hints,
                        test_data_nhwc_path=test_data_nhwc_path,
                        native_package_generation_timeout_sec=native_pytorch_generation_timeout_sec,
                        raise_on_failure=False,
                    )
                if pytorch_package_path is not None and output_dynamo_onnx_from_model_ir:
                    pytorch_dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
                        package_dir=str(pytorch_package_path),
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                        shape_hints=shape_hints,
                        test_data_nhwc_path=test_data_nhwc_path,
                        native_package_generation_timeout_sec=native_pytorch_generation_timeout_sec,
                        raise_on_failure=False,
                    )
                if pytorch_package_path is not None and output_exported_program_from_model_ir:
                    pytorch_exported_program_path = export_exported_program_from_generated_package(
                        package_dir=str(pytorch_package_path),
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                        shape_hints=shape_hints,
                        test_data_nhwc_path=test_data_nhwc_path,
                        native_package_generation_timeout_sec=native_pytorch_generation_timeout_sec,
                        raise_on_failure=False,
                    )
            _advance_export_progress()

        model_ir_fp32 = None

        _set_export_progress_desc("write float16 tflite")
        model_ir_fp16 = clone_model_ir_with_float16(model_ir)
        optimize_redundant_transpose_operators(
            model_ir_fp16,
            preserve_model_outputs=True,
        )
        _set_reduced_precision_support_metadata(
            model_ir=model_ir_fp16,
            enable_accumulation_type_float16=enable_accumulation_type_float16,
        )
        float16_path = os.path.join(output_folder_path, f"{output_file_name}_float16.tflite")
        float16_write_timing: Dict[str, Any] = {}
        model_ir_fp16_tflite = model_ir_fp16
        fp16_write_graph_index = ModelIRGraphIndex(model_ir_fp16_tflite)
        _optimize_constant_input_scatter_nd_chains(
            model_ir_fp16_tflite,
            graph_index=fp16_write_graph_index,
        )
        _optimize_constant_binary_elementwise_chains(
            model_ir_fp16_tflite,
            graph_index=fp16_write_graph_index,
        )
        run_model_ir_validation_pipeline(
            model_ir_fp16_tflite,
            graph_index=fp16_write_graph_index,
        )
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=model_ir_fp16_tflite,
            output_tflite_path=float16_path,
            timing=float16_write_timing,
        )
        write_timing_report["float16"] = float16_write_timing
        _progress_write(
            message=_format_write_timing_line(
                stage="float16",
                timing=float16_write_timing,
            ),
            enabled=bool(flatbuffer_direct_show_progress),
        )
        _advance_export_progress()
        del fp16_write_graph_index
        del model_ir_fp16_tflite
        del model_ir_fp16

        dynamic_range_path = None
        if output_dynamic_range_quantized_tflite:
            _set_export_progress_desc("write dynamic range quant tflite")
            dynamic_model_ir = build_dynamic_range_quantized_model_ir(
                model_ir,
                quant_type=str(quant_type),
                calibration_method=quant_controls["calibration_method"],
                calibration_percentile=quant_controls["calibration_percentile"],
                min_numel=quant_controls["min_numel"],
                min_abs_max=quant_controls["min_abs_max"],
                scale_floor=quant_controls["scale_floor"],
            )
            run_model_ir_validation_pipeline(dynamic_model_ir)
            dynamic_range_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_dynamic_range_quant.tflite",
            )
            dynamic_range_write_timing: Dict[str, Any] = {}
            write_model_file(
                schema_tflite=schema_tflite,
                model_ir=dynamic_model_ir,
                output_tflite_path=dynamic_range_path,
                timing=dynamic_range_write_timing,
            )
            write_timing_report["dynamic_range_quant"] = dynamic_range_write_timing
            _progress_write(
                message=_format_write_timing_line(
                    stage="dynamic_range_quant",
                    timing=dynamic_range_write_timing,
                ),
                enabled=bool(flatbuffer_direct_show_progress),
            )
            _advance_export_progress()
            del dynamic_model_ir

        integer_quant_path = None
        full_integer_quant_path = None
        integer_quant_with_int16_act_path = None
        full_integer_quant_with_int16_act_path = None
        int16_activation_skip_reasons: List[str] = []
        if output_integer_quantized_tflite:
            _set_export_progress_desc("write integer quant tflite")
            integer_result = build_integer_quantized_model_ir(
                model_ir,
                quant_type=str(quant_type),
                calibration_method=quant_controls["calibration_method"],
                calibration_percentile=quant_controls["calibration_percentile"],
                min_numel=quant_controls["min_numel"],
                min_abs_max=quant_controls["min_abs_max"],
                scale_floor=quant_controls["scale_floor"],
                calibration_ranges=calibration_ranges,
                return_report=True,
            )
            integer_model_ir = integer_result.model_ir
            calibration_report["variants"]["integer_quant"] = integer_result.report
            integer_quant_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_integer_quant.tflite",
            )
            integer_quant_write_timing: Dict[str, Any] = {}
            _write_validated_quantized_model_file(
                model_ir=integer_model_ir,
                output_tflite_path=integer_quant_path,
                timing=integer_quant_write_timing,
            )
            write_timing_report["integer_quant"] = integer_quant_write_timing
            _progress_write(
                message=_format_write_timing_line(
                    stage="integer_quant",
                    timing=integer_quant_write_timing,
                ),
                enabled=bool(flatbuffer_direct_show_progress),
            )
            _advance_export_progress()
            del integer_model_ir
            del integer_result

            _set_export_progress_desc("write full integer quant tflite")
            full_integer_result = build_full_integer_quantized_model_ir(
                model_ir,
                quant_type=str(quant_type),
                input_quant_dtype=str(input_quant_dtype),
                output_quant_dtype=str(output_quant_dtype),
                calibration_method=quant_controls["calibration_method"],
                calibration_percentile=quant_controls["calibration_percentile"],
                min_numel=quant_controls["min_numel"],
                min_abs_max=quant_controls["min_abs_max"],
                scale_floor=quant_controls["scale_floor"],
                calibration_ranges=calibration_ranges,
                return_report=True,
            )
            full_integer_model_ir = full_integer_result.model_ir
            calibration_report["variants"]["full_integer_quant"] = full_integer_result.report
            full_integer_quant_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_full_integer_quant.tflite",
            )
            full_integer_quant_write_timing: Dict[str, Any] = {}
            _write_validated_quantized_model_file(
                model_ir=full_integer_model_ir,
                output_tflite_path=full_integer_quant_path,
                timing=full_integer_quant_write_timing,
            )
            write_timing_report["full_integer_quant"] = full_integer_quant_write_timing
            _progress_write(
                message=_format_write_timing_line(
                    stage="full_integer_quant",
                    timing=full_integer_quant_write_timing,
                ),
                enabled=bool(flatbuffer_direct_show_progress),
            )
            _advance_export_progress()
            del full_integer_model_ir
            del full_integer_result

            int16_activation_skip_reasons = strict_int16_activation_skip_reasons(model_ir)
            if len(int16_activation_skip_reasons) > 0:
                calibration_report["variants"]["integer_quant_with_int16_act"] = {
                    "skipped": True,
                    "skip_reasons": list(int16_activation_skip_reasons),
                }
                calibration_report["variants"]["full_integer_quant_with_int16_act"] = {
                    "skipped": True,
                    "skip_reasons": list(int16_activation_skip_reasons),
                }
                _progress_write(
                    message=(
                        "integer_quant_with_int16_act skipped. "
                        + " ".join(int16_activation_skip_reasons)
                    ),
                    enabled=bool(flatbuffer_direct_show_progress),
                )
                _advance_export_progress()
                _progress_write(
                    message=(
                        "full_integer_quant_with_int16_act skipped. "
                        + " ".join(int16_activation_skip_reasons)
                    ),
                    enabled=bool(flatbuffer_direct_show_progress),
                )
                _advance_export_progress()
            else:
                _set_export_progress_desc("write integer quant int16-act tflite")
                integer_quant_with_int16_act_model_ir = build_integer_quantized_with_int16_act_model_ir(
                    model_ir,
                    quant_type=str(quant_type),
                    calibration_method=quant_controls["calibration_method"],
                    calibration_percentile=quant_controls["calibration_percentile"],
                    min_numel=quant_controls["min_numel"],
                    min_abs_max=quant_controls["min_abs_max"],
                    scale_floor=quant_controls["scale_floor"],
                    calibration_ranges=calibration_ranges,
                )
                integer_quant_with_int16_act_path = os.path.join(
                    output_folder_path,
                    f"{output_file_name}_integer_quant_with_int16_act.tflite",
                )
                integer_quant_int16_write_timing: Dict[str, Any] = {}
                _write_validated_quantized_model_file(
                    model_ir=integer_quant_with_int16_act_model_ir,
                    output_tflite_path=integer_quant_with_int16_act_path,
                    timing=integer_quant_int16_write_timing,
                )
                write_timing_report["integer_quant_with_int16_act"] = integer_quant_int16_write_timing
                _progress_write(
                    message=_format_write_timing_line(
                        stage="integer_quant_with_int16_act",
                        timing=integer_quant_int16_write_timing,
                    ),
                    enabled=bool(flatbuffer_direct_show_progress),
                )
                _advance_export_progress()
                del integer_quant_with_int16_act_model_ir

                _set_export_progress_desc("write full integer quant int16-act tflite")
                full_integer_quant_with_int16_act_model_ir = build_full_integer_quantized_with_int16_act_model_ir(
                    model_ir,
                    quant_type=str(quant_type),
                    calibration_method=quant_controls["calibration_method"],
                    calibration_percentile=quant_controls["calibration_percentile"],
                    min_numel=quant_controls["min_numel"],
                    min_abs_max=quant_controls["min_abs_max"],
                    scale_floor=quant_controls["scale_floor"],
                    calibration_ranges=calibration_ranges,
                )
                full_integer_quant_with_int16_act_path = os.path.join(
                    output_folder_path,
                    f"{output_file_name}_full_integer_quant_with_int16_act.tflite",
                )
                full_integer_quant_int16_write_timing: Dict[str, Any] = {}
                _write_validated_quantized_model_file(
                    model_ir=full_integer_quant_with_int16_act_model_ir,
                    output_tflite_path=full_integer_quant_with_int16_act_path,
                    timing=full_integer_quant_int16_write_timing,
                )
                write_timing_report["full_integer_quant_with_int16_act"] = full_integer_quant_int16_write_timing
                _progress_write(
                    message=_format_write_timing_line(
                        stage="full_integer_quant_with_int16_act",
                        timing=full_integer_quant_int16_write_timing,
                    ),
                    enabled=bool(flatbuffer_direct_show_progress),
                )
                _advance_export_progress()
                del full_integer_quant_with_int16_act_model_ir
            calibration_report_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_strict_integer_quant_calibration_report.json",
            )
            with open(calibration_report_path, "w", encoding="utf-8") as f:
                json.dump(calibration_report, f, indent=2)
            del calibration_ranges
            del calibration_report

        if output_weights:
            _set_export_progress_desc("export float32 weights")
            weights_export(
                extract_target_tflite_file_path=float32_path,
                output_weights_file_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_float32_weights.h5",
                ),
            )
            _advance_export_progress()

            _set_export_progress_desc("export float16 weights")
            weights_export(
                extract_target_tflite_file_path=float16_path,
                output_weights_file_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_float16_weights.h5",
                ),
            )
            _advance_export_progress()

            if dynamic_range_path is not None:
                _set_export_progress_desc("export dynamic range quant weights")
                weights_export(
                    extract_target_tflite_file_path=dynamic_range_path,
                    output_weights_file_path=os.path.join(
                        output_folder_path,
                        f"{output_file_name}_dynamic_range_quant_weights.h5",
                    ),
                )
                _advance_export_progress()

            if integer_quant_path is not None:
                _set_export_progress_desc("export integer quant weights")
                weights_export(
                    extract_target_tflite_file_path=integer_quant_path,
                    output_weights_file_path=os.path.join(
                        output_folder_path,
                        f"{output_file_name}_integer_quant_weights.h5",
                    ),
                )
                _advance_export_progress()

            if full_integer_quant_path is not None:
                _set_export_progress_desc("export full integer quant weights")
                weights_export(
                    extract_target_tflite_file_path=full_integer_quant_path,
                    output_weights_file_path=os.path.join(
                        output_folder_path,
                        f"{output_file_name}_full_integer_quant_weights.h5",
                    ),
                )
                _advance_export_progress()

            if integer_quant_with_int16_act_path is not None:
                _set_export_progress_desc("export integer quant int16-act weights")
                weights_export(
                    extract_target_tflite_file_path=integer_quant_with_int16_act_path,
                    output_weights_file_path=os.path.join(
                        output_folder_path,
                        f"{output_file_name}_integer_quant_with_int16_act_weights.h5",
                    ),
                )
                _advance_export_progress()

            if full_integer_quant_with_int16_act_path is not None:
                _set_export_progress_desc("export full integer quant int16-act weights")
                weights_export(
                    extract_target_tflite_file_path=full_integer_quant_with_int16_act_path,
                    output_weights_file_path=os.path.join(
                        output_folder_path,
                        f"{output_file_name}_full_integer_quant_with_int16_act_weights.h5",
                    ),
                )
                _advance_export_progress()
    finally:
        export_progress_spinner.stop()
        if export_progress_bar is not None:
            export_progress_bar.close()

    outputs: Dict[str, Any] = {
        "float32_tflite_path": float32_path,
        "float16_tflite_path": float16_path,
    }
    if saved_model_path is not None:
        outputs["saved_model_path"] = str(saved_model_path)
        outputs["saved_model_persisted"] = bool(persist_saved_model_output)
    if pytorch_package_path is not None:
        outputs["pytorch_package_path"] = str(pytorch_package_path)
    if pytorch_torchscript_path is not None:
        outputs["pytorch_torchscript_path"] = str(pytorch_torchscript_path)
    if pytorch_dynamo_onnx_path is not None:
        outputs["pytorch_dynamo_onnx_path"] = str(pytorch_dynamo_onnx_path)
    if pytorch_exported_program_path is not None:
        outputs["pytorch_exported_program_path"] = str(pytorch_exported_program_path)
    if len(write_timing_report) > 0:
        outputs["write_timing_report"] = write_timing_report
    if dynamic_range_path is not None:
        outputs["dynamic_range_quant_tflite_path"] = dynamic_range_path
    if integer_quant_path is not None:
        outputs["integer_quant_tflite_path"] = integer_quant_path
    if full_integer_quant_path is not None:
        outputs["full_integer_quant_tflite_path"] = full_integer_quant_path
    if integer_quant_with_int16_act_path is not None:
        outputs["integer_quant_with_int16_act_tflite_path"] = integer_quant_with_int16_act_path
    if full_integer_quant_with_int16_act_path is not None:
        outputs["full_integer_quant_with_int16_act_tflite_path"] = full_integer_quant_with_int16_act_path
    if len(int16_activation_skip_reasons) > 0:
        outputs["int16_activation_quantization_skipped"] = True
        outputs["int16_activation_quantization_skip_reasons"] = list(int16_activation_skip_reasons)
    if calibration_report_path is not None:
        outputs["strict_integer_quant_calibration_report_path"] = calibration_report_path
    if split_plan_report_path is not None:
        outputs["split_plan_report_path"] = split_plan_report_path
        outputs["split_required_by_estimate"] = bool(split_required_by_estimate)
        if split_plan_total_estimated_bytes is not None:
            outputs["split_plan_total_estimated_bytes"] = int(split_plan_total_estimated_bytes)
    if split_manifest_path is not None:
        outputs["split_manifest_path"] = split_manifest_path
    if split_partition_paths is not None:
        outputs["split_partition_paths"] = split_partition_paths
        outputs["split_partition_count"] = int(split_partition_count)
    if split_saved_model_dirs is not None:
        outputs["split_saved_model_dirs"] = list(split_saved_model_dirs)
        outputs["split_saved_model_count"] = int(len(split_saved_model_dirs))
    if split_pytorch_package_dirs is not None:
        outputs["split_pytorch_package_dirs"] = list(split_pytorch_package_dirs)
        outputs["split_pytorch_package_count"] = int(len(split_pytorch_package_dirs))
    if split_pytorch_torchscript_paths is not None:
        outputs["split_pytorch_torchscript_paths"] = list(split_pytorch_torchscript_paths)
        outputs["split_pytorch_torchscript_count"] = int(len(split_pytorch_torchscript_paths))
    if split_pytorch_dynamo_onnx_paths is not None:
        outputs["split_pytorch_dynamo_onnx_paths"] = list(split_pytorch_dynamo_onnx_paths)
        outputs["split_pytorch_dynamo_onnx_count"] = int(len(split_pytorch_dynamo_onnx_paths))
    if split_pytorch_exported_program_paths is not None:
        outputs["split_pytorch_exported_program_paths"] = list(split_pytorch_exported_program_paths)
        outputs["split_pytorch_exported_program_count"] = int(len(split_pytorch_exported_program_paths))
    if op_coverage_report_path is not None:
        outputs["op_coverage_report_path"] = op_coverage_report_path
    outputs["tensor_correspondence_report_path"] = tensor_correspondence_report_path
    outputs["custom_op_count"] = int(len(custom_ops_used))
    if len(custom_ops_used) > 0:
        outputs["custom_ops_used"] = custom_ops_used
    if len(custom_op_nodes) > 0:
        outputs["custom_op_nodes"] = custom_op_nodes
    return ConversionResult.from_legacy_dict(outputs).to_legacy_dict()
