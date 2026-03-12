from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.ir import (
    clone_model_ir_with_float16,
    clone_model_ir_with_float32,
    optimize_redundant_transpose_operators,
    prune_identity_cast_operators,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
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
    build_dynamic_range_quantized_model_ir,
    build_full_integer_quantized_model_ir,
    build_full_integer_quantized_with_int16_act_model_ir,
    build_integer_quantized_model_ir,
    build_integer_quantized_with_int16_act_model_ir,
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
from onnx2tf.tflite_builder.split_saved_model_exporter import (
    export_split_saved_models,
)
from onnx2tf.utils.common_functions import weights_export


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


def _create_progress_bar(
    *,
    total: int,
    desc: str,
    enabled: bool,
):
    if not enabled or int(total) <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(
        total=int(total),
        desc=str(desc),
        dynamic_ncols=True,
    )


class _ProgressSpinner:
    def __init__(self, progress_bar: Any) -> None:
        self._progress_bar = progress_bar
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.stop()
        if self._progress_bar is None:
            return
        self._stop_event = threading.Event()
        self._progress_bar.set_postfix_str("|", refresh=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=0.5)
        self._thread = None
        if self._progress_bar is not None:
            self._progress_bar.set_postfix_str("", refresh=True)

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        frame_index = 0
        while not self._stop_event.wait(0.1):
            if self._progress_bar is None:
                return
            frame_index = (frame_index + 1) % len(frames)
            self._progress_bar.set_postfix_str(frames[frame_index], refresh=True)


def _build_export_progress_labels(
    *,
    report_op_coverage: bool,
    split_plan_requested: bool,
    output_dynamic_range_quantized_tflite: bool,
    output_integer_quantized_tflite: bool,
    output_weights: bool,
    output_saved_model_from_model_ir: bool,
    output_pytorch_from_model_ir: bool,
) -> List[str]:
    labels: List[str] = [
        "tensor correspondence report",
    ]
    if report_op_coverage:
        labels.append("op coverage report")
    if split_plan_requested:
        labels.append("split planning")
    labels.extend(
        [
            "write float32 tflite",
            "write saved_model",
            "write pytorch",
            "write float16 tflite",
        ]
    )
    if not output_saved_model_from_model_ir:
        labels.remove("write saved_model")
    if not output_pytorch_from_model_ir:
        labels.remove("write pytorch")
    if output_dynamic_range_quantized_tflite:
        labels.append("write dynamic range quant tflite")
    if output_integer_quantized_tflite:
        labels.extend(
            [
                "write integer quant tflite",
                "write full integer quant tflite",
                "write integer quant int16-act tflite",
                "write full integer quant int16-act tflite",
            ]
        )
    if output_weights:
        labels.extend(
            [
                "export float32 weights",
                "export float16 weights",
            ]
        )
        if output_dynamic_range_quantized_tflite:
            labels.append("export dynamic range quant weights")
        if output_integer_quantized_tflite:
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


def _resolve_quantization_controls(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    calibration_method = kwargs.get(
        "flatbuffer_direct_calibration_method",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_METHOD", "max"),
    )
    calibration_percentile = kwargs.get(
        "flatbuffer_direct_calibration_percentile",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_PERCENTILE", "99.99"),
    )
    quant_min_numel = kwargs.get(
        "flatbuffer_direct_quant_min_numel",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_NUMEL", "1"),
    )
    quant_min_abs_max = kwargs.get(
        "flatbuffer_direct_quant_min_abs_max",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_ABS_MAX", "0.0"),
    )
    quant_scale_floor = kwargs.get(
        "flatbuffer_direct_quant_scale_floor",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_QUANT_SCALE_FLOOR", "1e-8"),
    )
    return {
        "calibration_method": str(calibration_method),
        "calibration_percentile": float(calibration_percentile),
        "min_numel": int(quant_min_numel),
        "min_abs_max": float(quant_min_abs_max),
        "scale_floor": float(quant_scale_floor),
    }


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
    _reject_unsupported_quantization(**kwargs)

    output_folder_path = kwargs.get("output_folder_path", "saved_model")
    output_file_name = kwargs.get("output_file_name", "model")
    onnx_graph = kwargs.get("onnx_graph", None)
    output_weights = bool(kwargs.get("output_weights", False))
    quant_type = kwargs.get("quant_type", "per-channel")
    input_quant_dtype = kwargs.get("input_quant_dtype", "int8")
    output_quant_dtype = kwargs.get("output_quant_dtype", "int8")
    output_dynamic_range_quantized_tflite = bool(
        kwargs.get("output_dynamic_range_quantized_tflite", False)
    )
    output_integer_quantized_tflite = bool(
        kwargs.get("output_integer_quantized_tflite", False)
    )
    output_saved_model_from_model_ir = bool(
        kwargs.get("output_saved_model_from_model_ir", False)
    )
    output_pytorch_from_model_ir = bool(
        kwargs.get("output_pytorch_from_model_ir", False)
    )
    output_torchscript_from_model_ir = bool(
        kwargs.get("output_torchscript_from_model_ir", False)
    )
    output_dynamo_onnx_from_model_ir = bool(
        kwargs.get("output_dynamo_onnx_from_model_ir", False)
    )
    output_exported_program_from_model_ir = bool(
        kwargs.get("output_exported_program_from_model_ir", False)
    )
    if (
        output_torchscript_from_model_ir
        or output_dynamo_onnx_from_model_ir
        or output_exported_program_from_model_ir
    ):
        output_pytorch_from_model_ir = True
    saved_model_output_folder_path = kwargs.get(
        "saved_model_output_folder_path",
        None,
    )
    if saved_model_output_folder_path is None:
        saved_model_output_folder_path = output_folder_path
    pytorch_output_folder_path = kwargs.get(
        "pytorch_output_folder_path",
        None,
    )
    if pytorch_output_folder_path is None:
        pytorch_output_folder_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_pytorch",
        )
    persist_saved_model_output = bool(
        kwargs.get(
            "persist_saved_model_output",
            output_saved_model_from_model_ir,
        )
    )
    enable_accumulation_type_float16 = bool(
        kwargs.get("enable_accumulation_type_float16", False)
    )
    force_split_manifest = bool(kwargs.get("force_split_manifest", False))
    split_plan_requested = bool(force_split_manifest)
    report_op_coverage = bool(kwargs.get("report_op_coverage", False))
    custom_input_op_name_np_data_path = kwargs.get(
        "custom_input_op_name_np_data_path",
        None,
    )
    shape_hints = kwargs.get("shape_hints", None)
    test_data_nhwc_path = kwargs.get("test_data_nhwc_path", None)
    output_nms_with_argmax = bool(kwargs.get("output_nms_with_argmax", False))
    switch_nms_version = str(kwargs.get("switch_nms_version", "v4")).strip().lower()
    if switch_nms_version not in {"v4", "v5"}:
        raise ValueError(
            "switch_nms_version must be 'v4' or 'v5'. "
            f"got: {switch_nms_version}"
        )
    keep_ncw_or_nchw_or_ncdhw_input_names = kwargs.get(
        "keep_ncw_or_nchw_or_ncdhw_input_names",
        None,
    )
    keep_nwc_or_nhwc_or_ndhwc_input_names = kwargs.get(
        "keep_nwc_or_nhwc_or_ndhwc_input_names",
        None,
    )
    keep_shape_absolutely_input_names = kwargs.get(
        "keep_shape_absolutely_input_names",
        None,
    )
    disable_group_convolution = bool(
        kwargs.get("disable_group_convolution", False)
    )
    enable_batchmatmul_unfold = bool(
        kwargs.get("enable_batchmatmul_unfold", False)
    )
    enable_rnn_unroll = bool(
        kwargs.get("enable_rnn_unroll", False)
    )
    mvn_epsilon = float(kwargs.get("mvn_epsilon", 1e-10))
    flatbuffer_direct_allow_custom_ops = bool(
        kwargs.get("flatbuffer_direct_allow_custom_ops", False)
    )
    custom_allowlist_raw = kwargs.get(
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
    tflite_split_max_bytes = int(
        kwargs.get(
            "tflite_split_max_bytes",
            os.environ.get(
                "ONNX2TF_FLATBUFFER_DIRECT_SPLIT_MAX_BYTES",
                str(DEFAULT_TFLITE_SPLIT_MAX_BYTES),
            ),
        )
    )
    tflite_split_target_bytes = int(
        kwargs.get(
            "tflite_split_target_bytes",
            os.environ.get(
                "ONNX2TF_FLATBUFFER_DIRECT_SPLIT_TARGET_BYTES",
                str(DEFAULT_TFLITE_SPLIT_TARGET_BYTES),
            ),
        )
    )
    quant_controls = _resolve_quantization_controls(kwargs)
    flatbuffer_direct_show_progress = bool(
        kwargs.get("flatbuffer_direct_show_progress", True)
    )
    number_of_dimensions_after_flextranspose_compression = int(
        kwargs.get("number_of_dimensions_after_flextranspose_compression", 6)
    )
    disable_suppression_flextranspose = bool(
        kwargs.get("disable_suppression_flextranspose", False)
    )
    number_of_dimensions_after_flexstridedslice_compression = int(
        kwargs.get("number_of_dimensions_after_flexstridedslice_compression", 5)
    )
    disable_suppression_flexstridedslice = bool(
        kwargs.get("disable_suppression_flexstridedslice", False)
    )
    optimization_for_gpu_delegate = bool(
        kwargs.get("optimization_for_gpu_delegate", False)
    )
    replace_argmax_to_reducemax_and_indices_is_int64 = bool(
        kwargs.get("replace_argmax_to_reducemax_and_indices_is_int64", False)
    )
    replace_argmax_to_reducemax_and_indices_is_float32 = bool(
        kwargs.get("replace_argmax_to_reducemax_and_indices_is_float32", False)
    )
    replace_argmax_to_fused_argmax_and_indices_is_int64 = bool(
        kwargs.get("replace_argmax_to_fused_argmax_and_indices_is_int64", False)
    )
    replace_argmax_to_fused_argmax_and_indices_is_float32 = bool(
        kwargs.get("replace_argmax_to_fused_argmax_and_indices_is_float32", False)
    )
    fused_argmax_scale_ratio = float(
        kwargs.get("fused_argmax_scale_ratio", 0.5)
    )
    requested_pseudo_ops_raw = kwargs.get("replace_to_pseudo_operators", None)
    input_names_to_interrupt_model_conversion = kwargs.get(
        "input_names_to_interrupt_model_conversion",
        None,
    )
    output_names_to_interrupt_model_conversion = kwargs.get(
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
            try:
                _write_coverage_report(str(ex))
            except Exception:
                pass
            raise
    finally:
        configure_pseudo_ops_wave1_targets(None)

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
        )
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
    except Exception as ex:
        try:
            _write_coverage_report(str(ex))
        except Exception:
            pass
        raise

    export_progress_labels = _build_export_progress_labels(
        report_op_coverage=report_op_coverage,
        split_plan_requested=split_plan_requested,
        output_dynamic_range_quantized_tflite=output_dynamic_range_quantized_tflite,
        output_integer_quantized_tflite=output_integer_quantized_tflite,
        output_weights=output_weights,
        output_saved_model_from_model_ir=output_saved_model_from_model_ir,
        output_pytorch_from_model_ir=output_pytorch_from_model_ir,
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
        else:
            _write_coverage_report(None)

        custom_ops_used = sorted(
            list(
                {
                    str(op.options.get("customCode", "CUSTOM"))
                    for op in model_ir.operators
                    if str(op.op_type) == "CUSTOM"
                }
            )
        )
        custom_op_nodes: List[Dict[str, str]] = []
        custom_op_nodes_seen = set()
        for op in model_ir.operators:
            if str(op.op_type) != "CUSTOM":
                continue
            options = op.options if isinstance(op.options, dict) else {}
            custom_code = str(options.get("customCode", "CUSTOM")).strip()
            if custom_code == "":
                custom_code = "CUSTOM"
            onnx_op = str(options.get("onnxOp", "")).strip()
            onnx_node_name = str(options.get("onnxNodeName", "")).strip()
            key = (custom_code, onnx_op, onnx_node_name)
            if key in custom_op_nodes_seen:
                continue
            custom_op_nodes_seen.add(key)
            custom_op_nodes.append(
                {
                    "custom_code": custom_code,
                    "onnx_op": onnx_op,
                    "onnx_node_name": onnx_node_name,
                }
            )
        custom_op_nodes.sort(
            key=lambda v: (
                str(v.get("custom_code", "")),
                str(v.get("onnx_op", "")),
                str(v.get("onnx_node_name", "")),
            )
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
        if split_plan_requested:
            _set_export_progress_desc("split planning")
            split_plan_report = plan_contiguous_partitions_by_size(
                model_ir=model_ir,
                target_max_bytes=tflite_split_target_bytes,
                hard_max_bytes=tflite_split_max_bytes,
                schema_tflite=schema_tflite,
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
                )
                split_manifest_path = split_outputs["split_manifest_path"]
                split_partition_paths = split_outputs["split_partition_paths"]
                split_partition_count = int(split_outputs["split_partition_count"])
            _advance_export_progress()

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
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=model_ir_fp32,
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

        if output_saved_model_from_model_ir:
            _set_export_progress_desc("write saved_model")
            if split_manifest_path is not None:
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

                pytorch_package_path = export_pytorch_package_from_model_ir(
                    model_ir=model_ir_fp32,
                    output_folder_path=pytorch_output_folder_path,
                    fallback_tflite_path=float32_path,
                    fallback_onnx_graph=onnx_graph,
                    fallback_saved_model_path=fallback_saved_model_path,
                    fallback_saved_model_factory=_ensure_pytorch_saved_model_bridge,
                    fallback_tflite_has_custom_ops=bool(len(custom_ops_used) > 0),
                )
                if output_torchscript_from_model_ir:
                    pytorch_torchscript_path = export_torchscript_from_generated_package(
                        package_dir=str(pytorch_package_path),
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                        shape_hints=shape_hints,
                        test_data_nhwc_path=test_data_nhwc_path,
                        raise_on_failure=False,
                    )
                if output_dynamo_onnx_from_model_ir:
                    pytorch_dynamo_onnx_path = export_dynamo_onnx_from_generated_package(
                        package_dir=str(pytorch_package_path),
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                        shape_hints=shape_hints,
                        test_data_nhwc_path=test_data_nhwc_path,
                        raise_on_failure=False,
                    )
                if output_exported_program_from_model_ir:
                    pytorch_exported_program_path = export_exported_program_from_generated_package(
                        package_dir=str(pytorch_package_path),
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                        shape_hints=shape_hints,
                        test_data_nhwc_path=test_data_nhwc_path,
                        raise_on_failure=False,
                    )
            _advance_export_progress()

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
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=model_ir_fp16,
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

        integer_quant_path = None
        full_integer_quant_path = None
        if output_integer_quantized_tflite:
            _set_export_progress_desc("write integer quant tflite")
            integer_model_ir = build_integer_quantized_model_ir(
                model_ir,
                quant_type=str(quant_type),
                calibration_method=quant_controls["calibration_method"],
                calibration_percentile=quant_controls["calibration_percentile"],
                min_numel=quant_controls["min_numel"],
                min_abs_max=quant_controls["min_abs_max"],
                scale_floor=quant_controls["scale_floor"],
            )
            integer_quant_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_integer_quant.tflite",
            )
            integer_quant_write_timing: Dict[str, Any] = {}
            write_model_file(
                schema_tflite=schema_tflite,
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

            _set_export_progress_desc("write full integer quant tflite")
            full_integer_model_ir = build_full_integer_quantized_model_ir(
                model_ir,
                quant_type=str(quant_type),
                input_quant_dtype=str(input_quant_dtype),
                output_quant_dtype=str(output_quant_dtype),
                calibration_method=quant_controls["calibration_method"],
                calibration_percentile=quant_controls["calibration_percentile"],
                min_numel=quant_controls["min_numel"],
                min_abs_max=quant_controls["min_abs_max"],
                scale_floor=quant_controls["scale_floor"],
            )
            full_integer_quant_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_full_integer_quant.tflite",
            )
            full_integer_quant_write_timing: Dict[str, Any] = {}
            write_model_file(
                schema_tflite=schema_tflite,
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

            _set_export_progress_desc("write integer quant int16-act tflite")
            integer_quant_with_int16_act_model_ir = build_integer_quantized_with_int16_act_model_ir(
                model_ir,
                quant_type=str(quant_type),
                calibration_method=quant_controls["calibration_method"],
                calibration_percentile=quant_controls["calibration_percentile"],
                min_numel=quant_controls["min_numel"],
                min_abs_max=quant_controls["min_abs_max"],
                scale_floor=quant_controls["scale_floor"],
            )
            integer_quant_with_int16_act_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_integer_quant_with_int16_act.tflite",
            )
            integer_quant_int16_write_timing: Dict[str, Any] = {}
            write_model_file(
                schema_tflite=schema_tflite,
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

            _set_export_progress_desc("write full integer quant int16-act tflite")
            full_integer_quant_with_int16_act_model_ir = build_full_integer_quantized_with_int16_act_model_ir(
                model_ir,
                quant_type=str(quant_type),
                calibration_method=quant_controls["calibration_method"],
                calibration_percentile=quant_controls["calibration_percentile"],
                min_numel=quant_controls["min_numel"],
                min_abs_max=quant_controls["min_abs_max"],
                scale_floor=quant_controls["scale_floor"],
            )
            full_integer_quant_with_int16_act_path = os.path.join(
                output_folder_path,
                f"{output_file_name}_full_integer_quant_with_int16_act.tflite",
            )
            full_integer_quant_int16_write_timing: Dict[str, Any] = {}
            write_model_file(
                schema_tflite=schema_tflite,
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
        else:
            integer_quant_with_int16_act_path = None
            full_integer_quant_with_int16_act_path = None

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
    return outputs
