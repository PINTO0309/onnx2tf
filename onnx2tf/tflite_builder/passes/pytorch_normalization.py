from __future__ import annotations

import copy
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
from onnx2tf.tflite_builder.passes.pytorch_layout_validation import (
    _align_public_boundary_shapes_to_onnx_contract,
    _apply_feature_last_sequence_layouts,
    _collect_feature_last_sequence_tensor_names,
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
    _repair_orphan_recurrent_step_tensors,
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


def normalize_model_ir_for_pytorch_channel_first(model_ir: ModelIR) -> ModelIR:
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
    return normalized
