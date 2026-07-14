from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    channel_first_logical_layout,
    channel_last_logical_layout,
    infer_model_ir_logical_layouts,
    is_channel_last_logical_layout,
    normalize_logical_layout,
    validate_model_ir_layout_annotations,
)
from onnx2tf.tflite_builder.passes.pytorch_compat import (
    _reject_residual_layout_transposes,
    _remove_redundant_layout_transposes,
    _rewrite_atan2_ones_like_to_atan,
)
from onnx2tf.tflite_builder.passes.pytorch_control_flow import (
    _rewrite_counter_bounded_while_ops_for_native_export,
    _rewrite_static_while_ops_for_native_export,
)
from onnx2tf.tflite_builder.passes.pytorch_layout_validation import (
    _align_public_boundary_shapes_to_onnx_contract,
    _apply_feature_last_sequence_layouts,
    _collect_feature_last_sequence_tensor_names,
    _ensure_public_boundary_layout_bridges,
    _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island,
    _propagate_pytorch_friendly_layouts,
    _restore_non_preserved_channel_first_layouts,
    _rewrite_filter_tensors_for_pytorch,
    _rewrite_layout_sensitive_ops,
    _shrink_preserved_channel_last_regions_for_pytorch,
    _synchronize_reshape_targets_with_output_tensors,
    validate_channel_first_exportability,
)
from onnx2tf.tflite_builder.passes.pytorch_recurrent import (
    _NATIVE_PYTORCH_RECURRENT_OP_TYPES,
    _repair_orphan_recurrent_step_tensors,
    _rewrite_recurrent_ops_for_native_export,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _collect_kernel_weight_tensor_names,
    _perm_cf_to_cl,
    _permute_tensor_to_channel_first_inplace,
    _read_transpose_perm,
)


@dataclass(frozen=True)
class _PyTorchNormalizationResult:
    model_ir: ModelIR
    graph_index: ModelIRGraphIndex


def _normalize_model_ir_for_pytorch_channel_first_with_index(
    model_ir: ModelIR,
) -> _PyTorchNormalizationResult:
    normalized = copy.deepcopy(model_ir)
    original_public_boundary_shapes: Dict[str, List[int]] = {}
    original_public_boundary_layouts: Dict[str, str] = {}
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        original_public_boundary_shapes[str(tensor_name)] = [
            int(value)
            for value in list(tensor.shape_signature or tensor.shape)
        ]
        original_public_boundary_layouts[str(tensor_name)] = (
            normalize_logical_layout(tensor.logical_layout)
        )
    infer_model_ir_logical_layouts(normalized)
    layout_graph_index = ModelIRGraphIndex(normalized)
    producer_index = layout_graph_index.producers
    consumer_index = layout_graph_index.consumers
    preserve_channel_last_tensor_names = (
        _collect_feature_last_sequence_tensor_names(
            normalized,
            graph_index=layout_graph_index,
        )
    )
    feature_last_changed = _apply_feature_last_sequence_layouts(
        normalized,
        preserve_channel_last_tensor_names,
        consumers=consumer_index,
    )
    if feature_last_changed:
        infer_model_ir_logical_layouts(normalized)
    shrunken_preserve_channel_last_tensor_names = (
        _shrink_preserved_channel_last_regions_for_pytorch(
            normalized,
            preserve_channel_last_tensor_names,
            producer_index=producer_index,
            consumers=consumer_index,
        )
    )
    preserve_names_changed = (
        shrunken_preserve_channel_last_tensor_names
        != preserve_channel_last_tensor_names
    )
    preserve_channel_last_tensor_names = (
        shrunken_preserve_channel_last_tensor_names
    )
    feature_last_changed = _apply_feature_last_sequence_layouts(
        normalized,
        preserve_channel_last_tensor_names,
        consumers=consumer_index,
    )
    if preserve_names_changed or feature_last_changed:
        infer_model_ir_logical_layouts(normalized)
    annotation_problems = validate_model_ir_layout_annotations(normalized)
    if len(annotation_problems) > 0:
        raise ModelIRPyTorchExportError(
            "Channel-first normalization failed: invalid semantic layout "
            f"annotations. problems={annotation_problems}"
        )
    original_layouts = {
        str(name): normalize_logical_layout(tensor.logical_layout)
        for name, tensor in normalized.tensors.items()
    }
    _rewrite_layout_sensitive_ops(
        normalized,
        original_layouts,
        preserve_channel_last_tensor_names,
        graph_index=layout_graph_index,
    )
    _propagate_pytorch_friendly_layouts(
        normalized,
        graph_index=layout_graph_index,
    )
    kernel_weight_tensor_names = _collect_kernel_weight_tensor_names(normalized)
    for tensor_name, tensor in normalized.tensors.items():
        if str(tensor_name) in kernel_weight_tensor_names:
            continue
        if str(tensor_name) in preserve_channel_last_tensor_names:
            continue
        _permute_tensor_to_channel_first_inplace(tensor)
    _synchronize_reshape_targets_with_output_tensors(
        normalized,
        preserve_channel_last_tensor_names,
        graph_index=layout_graph_index,
    )
    _rewrite_filter_tensors_for_pytorch(
        normalized,
        graph_index=layout_graph_index,
    )
    _remove_redundant_layout_transposes(
        normalized,
        original_layouts,
        preserve_channel_last_tensor_names,
        graph_index=layout_graph_index,
    )
    _propagate_pytorch_friendly_layouts(
        normalized,
        graph_index=layout_graph_index,
    )
    _apply_feature_last_sequence_layouts(
        normalized,
        preserve_channel_last_tensor_names,
        consumers=layout_graph_index.consumers,
    )
    _restore_non_preserved_channel_first_layouts(
        normalized,
        preserve_channel_last_tensor_names,
    )
    _rewrite_atan2_ones_like_to_atan(
        normalized,
        graph_index=layout_graph_index,
    )
    _repair_orphan_recurrent_step_tensors(
        normalized,
        graph_index=layout_graph_index,
    )
    public_layout_map = normalized.metadata.get("onnx_public_layout_map", None)
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
        normalized.metadata["onnx_public_layout_map"] = public_layout_map
    boundary_shape_map = normalized.metadata.get(
        "onnx_boundary_shape_signature_map",
        None,
    )
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
        normalized.metadata["onnx_boundary_shape_signature_map"] = (
            boundary_shape_map
        )
    preserve_public_channel_last_boundaries = (
        _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(
            normalized
        )
    )
    for tensor_name in list(normalized.inputs) + list(normalized.outputs):
        normalized_tensor_name = str(tensor_name)
        original_layout = original_public_boundary_layouts.get(
            normalized_tensor_name,
            LOGICAL_LAYOUT_UNKNOWN,
        )
        if (
            original_layout in {"NWC", "NHWC", "NDHWC"}
            and (
                preserve_public_channel_last_boundaries
                or normalized_tensor_name
                in preserve_channel_last_tensor_names
            )
            and normalized_tensor_name in original_public_boundary_shapes
        ):
            public_layout_map[normalized_tensor_name] = original_layout
            boundary_shape_map[normalized_tensor_name] = list(
                original_public_boundary_shapes[normalized_tensor_name]
            )
    for output_name in list(normalized.outputs):
        normalized_output_name = str(output_name)
        if normalized_output_name not in preserve_channel_last_tensor_names:
            continue
        output_tensor = normalized.tensors.get(normalized_output_name, None)
        if output_tensor is None:
            continue
        output_rank = len(list(output_tensor.shape))
        if output_rank in {3, 4, 5}:
            public_layout_map[normalized_output_name] = (
                channel_last_logical_layout(output_rank)
            )
    public_outputs = {str(value) for value in normalized.outputs}
    for op_index in layout_graph_index.operator_indices("TRANSPOSE"):
        op = normalized.operators[int(op_index)]
        if len(op.inputs) < 1 or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        if output_name not in public_outputs:
            continue
        if output_name in preserve_channel_last_tensor_names:
            continue
        output_tensor = normalized.tensors.get(output_name, None)
        input_tensor = normalized.tensors.get(str(op.inputs[0]), None)
        if output_tensor is None or input_tensor is None:
            continue
        output_rank = len(list(output_tensor.shape))
        if output_rank != 3:
            continue
        if _read_transpose_perm(normalized, op) != _perm_cf_to_cl(output_rank):
            continue
        if is_channel_last_logical_layout(
            normalize_logical_layout(input_tensor.logical_layout)
        ):
            public_layout_map[output_name] = channel_first_logical_layout(
                output_rank
            )
    _align_public_boundary_shapes_to_onnx_contract(
        normalized,
        graph_index=layout_graph_index,
    )
    normalized.metadata["assume_channel_last_layout_tensor_names"] = []
    _reject_residual_layout_transposes(
        normalized,
        preserve_channel_last_tensor_names,
        graph_index=layout_graph_index,
    )
    validate_channel_first_exportability(
        normalized,
        preserve_channel_last_tensor_names,
        graph_index=layout_graph_index,
    )
    return _PyTorchNormalizationResult(
        model_ir=normalized,
        graph_index=layout_graph_index,
    )


def normalize_model_ir_for_pytorch_channel_first(model_ir: ModelIR) -> ModelIR:
    return _normalize_model_ir_for_pytorch_channel_first_with_index(
        model_ir
    ).model_ir


def _collect_model_op_types(model_ir: ModelIR) -> set[str]:
    op_types = {str(op.op_type) for op in model_ir.operators}
    for subgraph in model_ir.subgraphs:
        op_types.update(_collect_model_op_types(subgraph))
    return op_types


def _is_layout_agnostic_native_model_ir(model_ir: ModelIR) -> bool:
    channel_sensitive_ops = {
        "AVERAGE_POOL_2D",
        "CONV_2D",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
        "DEPTHWISE_CONV_2D",
        "MAX_POOL_2D",
        "NON_MAX_SUPPRESSION_V4",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "TRANSPOSE_CONV",
    }
    return len(_collect_model_op_types(model_ir) & channel_sensitive_ops) == 0


def _rewrite_native_pytorch_compatibility_ops(model_ir: ModelIR) -> ModelIR:
    rewritten_model_ir = model_ir
    root_op_types = {str(op.op_type) for op in model_ir.operators}
    if "WHILE" in root_op_types:
        rewritten_model_ir = _rewrite_static_while_ops_for_native_export(
            rewritten_model_ir
        )
        rewritten_model_ir = _rewrite_counter_bounded_while_ops_for_native_export(
            rewritten_model_ir
        )
        root_op_types = {
            str(op.op_type) for op in rewritten_model_ir.operators
        }
    if root_op_types & _NATIVE_PYTORCH_RECURRENT_OP_TYPES:
        rewritten_model_ir = _rewrite_recurrent_ops_for_native_export(
            rewritten_model_ir
        )
    return rewritten_model_ir


def prepare_model_ir_for_native_pytorch(model_ir: ModelIR) -> ModelIR:
    original_boundary_shape_map = model_ir.metadata.get(
        "onnx_boundary_shape_signature_map",
        {},
    )
    if not isinstance(original_boundary_shape_map, dict):
        original_boundary_shape_map = {}
    original_public_layout_map = model_ir.metadata.get(
        "onnx_public_layout_map",
        {},
    )
    if not isinstance(original_public_layout_map, dict):
        original_public_layout_map = {}
    original_public_boundary_shapes: Dict[str, List[int]] = {}
    original_public_boundary_layouts: Dict[str, str] = {}
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        boundary_shape = original_boundary_shape_map.get(
            str(tensor_name),
            tensor.shape_signature or tensor.shape,
        )
        original_public_boundary_shapes[str(tensor_name)] = [
            int(value) for value in list(boundary_shape)
        ]
        explicit_public_layout = original_public_layout_map.get(
            str(tensor_name),
            None,
        )
        original_public_boundary_layouts[str(tensor_name)] = (
            normalize_logical_layout(
                explicit_public_layout
                if explicit_public_layout is not None
                else tensor.logical_layout
            )
        )
    rewritten_model_ir = _rewrite_native_pytorch_compatibility_ops(model_ir)
    try:
        normalization_result = _normalize_model_ir_for_pytorch_channel_first_with_index(
            rewritten_model_ir
        )
        prepared = normalization_result.model_ir
        boundary_graph_index = normalization_result.graph_index
    except ModelIRPyTorchExportError:
        if not _is_layout_agnostic_native_model_ir(rewritten_model_ir):
            raise
        prepared = copy.deepcopy(rewritten_model_ir)
        infer_model_ir_logical_layouts(prepared)
        prepared.metadata["assume_channel_last_layout_tensor_names"] = []
        boundary_graph_index = ModelIRGraphIndex(prepared)
    prepared_public_layout_map = {
        str(name): str(layout)
        for name, layout in original_public_boundary_layouts.items()
        if layout != LOGICAL_LAYOUT_UNKNOWN
    }
    prepared.metadata["onnx_public_layout_map"] = prepared_public_layout_map
    prepared.metadata["onnx_boundary_shape_signature_map"] = {
        str(name): [int(value) for value in list(shape)]
        for name, shape in original_public_boundary_shapes.items()
        if str(name) in prepared_public_layout_map
        or str(name) in original_public_layout_map
    }
    _ensure_public_boundary_layout_bridges(
        model_ir=prepared,
        desired_public_shape_map=original_public_boundary_shapes,
        desired_public_layout_map=original_public_boundary_layouts,
        graph_index=boundary_graph_index,
    )
    _align_public_boundary_shapes_to_onnx_contract(
        prepared,
        graph_index=boundary_graph_index,
    )
    return prepared
