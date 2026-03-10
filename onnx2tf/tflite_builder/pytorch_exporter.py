from __future__ import annotations

import copy
import json
import keyword
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
    channel_first_logical_layout,
    channel_last_logical_layout,
    infer_model_ir_logical_layouts,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
    rewrite_axis_for_layout,
    validate_model_ir_layout_annotations,
)
from onnx2tf.tflite_builder.pytorch_package_runtime import (
    SUPPORTED_TORCH_KERNEL_OP_TYPES,
)
from onnx2tf.tflite_builder.tflite_importer import (
    import_model_ir_from_tflite,
)


class ModelIRPyTorchExportError(RuntimeError):
    pass


def get_supported_pytorch_kernel_op_types() -> Set[str]:
    return set(SUPPORTED_TORCH_KERNEL_OP_TYPES)


def _perm_cl_to_cf(rank: int) -> Optional[List[int]]:
    if rank == 3:
        return [0, 2, 1]
    if rank == 4:
        return [0, 3, 1, 2]
    if rank == 5:
        return [0, 4, 1, 2, 3]
    return None


def _perm_cf_to_cl(rank: int) -> Optional[List[int]]:
    if rank == 3:
        return [0, 2, 1]
    if rank == 4:
        return [0, 2, 3, 1]
    if rank == 5:
        return [0, 2, 3, 4, 1]
    return None


def _permute_shape(values: Optional[Sequence[int]], perm: Sequence[int]) -> Optional[List[int]]:
    if values is None:
        return None
    items = [int(v) for v in list(values)]
    if len(items) != len(list(perm)):
        return None
    return [int(items[idx]) for idx in perm]


def _clone_tensor(tensor: TensorIR) -> TensorIR:
    return TensorIR(
        name=str(tensor.name),
        dtype=str(tensor.dtype),
        shape=[int(v) for v in list(tensor.shape)],
        shape_signature=(
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else None
        ),
        data=np.asarray(tensor.data).copy() if isinstance(tensor.data, np.ndarray) else tensor.data,
        is_variable=bool(tensor.is_variable),
        quantization=copy.deepcopy(tensor.quantization),
        logical_layout=normalize_logical_layout(tensor.logical_layout),
    )


def _read_transpose_perm(model_ir: ModelIR, op: OperatorIR) -> Optional[List[int]]:
    perm_tensor = model_ir.tensors.get(str(op.inputs[1]), None) if len(op.inputs) >= 2 else None
    if perm_tensor is not None and isinstance(perm_tensor.data, np.ndarray):
        perm = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
        if sorted(perm) == list(range(len(perm))):
            return perm
    perm = [int(v) for v in list(op.options.get("perm", []))]
    if len(perm) > 0 and sorted(perm) == list(range(len(perm))):
        return perm
    return None




def _rewrite_vector_constant_inplace(
    *,
    tensor: TensorIR,
    perm: Sequence[int],
    expected_rank: int,
) -> bool:
    if not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.ndim != 1 or int(arr.size) != int(expected_rank):
        return False
    tensor.data = np.asarray([arr[int(idx)] for idx in perm], dtype=arr.dtype)
    tensor.shape = [int(expected_rank)]
    if tensor.shape_signature is not None and len(tensor.shape_signature) == 1:
        tensor.shape_signature = [int(expected_rank)]
    return True


def _rewrite_matrix_constant_inplace(
    *,
    tensor: TensorIR,
    perm: Sequence[int],
    expected_rank: int,
) -> bool:
    if not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.ndim != 2 or tuple(arr.shape) != (int(expected_rank), 2):
        return False
    tensor.data = np.asarray(arr[list(perm), :], dtype=arr.dtype)
    tensor.shape = [int(expected_rank), 2]
    if tensor.shape_signature is not None and len(tensor.shape_signature) == 2:
        tensor.shape_signature = [int(expected_rank), 2]
    return True


def _rewrite_axis_constant_inplace(
    *,
    tensor: TensorIR,
    source_layout: str,
    target_layout: str,
    rank: int,
) -> bool:
    if not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.ndim == 0:
        axis = int(arr.reshape(-1)[0])
        rewritten = rewrite_axis_for_layout(
            axis=axis,
            source_layout=source_layout,
            target_layout=target_layout,
            rank=rank,
        )
        tensor.data = np.asarray(rewritten, dtype=arr.dtype)
        return True
    if arr.ndim != 1:
        return False
    rewritten_axes = [
        rewrite_axis_for_layout(
            axis=int(v),
            source_layout=source_layout,
            target_layout=target_layout,
            rank=rank,
        )
        for v in arr.reshape(-1).tolist()
    ]
    tensor.data = np.asarray(rewritten_axes, dtype=arr.dtype)
    tensor.shape = [int(len(rewritten_axes))]
    tensor.shape_signature = [int(len(rewritten_axes))]
    return True


def _permute_tensor_to_channel_first_inplace(tensor: TensorIR) -> bool:
    source_layout = normalize_logical_layout(tensor.logical_layout)
    rank = len(list(tensor.shape))
    if not is_channel_last_logical_layout(source_layout):
        return False
    target_layout = channel_first_logical_layout(rank)
    perm = logical_layout_permutation(
        source_layout=source_layout,
        target_layout=target_layout,
    )
    if perm is None:
        return False
    permuted_shape = _permute_shape(tensor.shape, perm)
    if permuted_shape is not None:
        tensor.shape = permuted_shape
    if tensor.shape_signature is not None:
        permuted_signature = _permute_shape(tensor.shape_signature, perm)
        if permuted_signature is not None:
            tensor.shape_signature = permuted_signature
    if isinstance(tensor.data, np.ndarray) and int(np.asarray(tensor.data).ndim) == int(rank):
        tensor.data = np.transpose(np.asarray(tensor.data), axes=perm).copy()
    tensor.logical_layout = target_layout
    return True


def _collect_kernel_weight_tensor_names(model_ir: ModelIR) -> Set[str]:
    names: Set[str] = set()
    for op in model_ir.operators:
        if str(op.op_type) in {
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "TRANSPOSE_CONV",
            "CONV_3D",
            "CONV_3D_TRANSPOSE",
        } and len(op.inputs) >= 2:
            names.add(str(op.inputs[1]))
    return names


def _primary_data_input_name(op: OperatorIR) -> Optional[str]:
    op_type = str(op.op_type)
    if len(op.inputs) == 0:
        return None
    if op_type == "SPLIT":
        return str(op.inputs[1]) if len(op.inputs) >= 2 else str(op.inputs[0])
    if op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"}:
        return str(op.inputs[2]) if len(op.inputs) >= 3 else None
    return str(op.inputs[0])


def _rewrite_layout_sensitive_ops(model_ir: ModelIR, original_layouts: Dict[str, str]) -> None:
    for op in model_ir.operators:
        op_type = str(op.op_type)
        data_input_name = _primary_data_input_name(op)
        data_tensor = model_ir.tensors.get(str(data_input_name), None) if data_input_name is not None else None
        if data_tensor is None:
            continue
        original_layout = normalize_logical_layout(original_layouts.get(str(data_input_name), data_tensor.logical_layout))
        rank = len(list(data_tensor.shape))
        if rank not in {3, 4, 5} or not is_channel_last_logical_layout(original_layout):
            continue
        target_layout = channel_first_logical_layout(rank)

        if op_type in {"CONCATENATION", "PACK", "UNPACK", "GATHER", "SOFTMAX", "ARG_MAX", "ARG_MIN"}:
            axis = op.options.get("axis", None)
            if axis is not None:
                op.options["axis"] = rewrite_axis_for_layout(
                    axis=int(axis),
                    source_layout=original_layout,
                    target_layout=target_layout,
                    rank=rank,
                )
        elif op_type == "SPLIT":
            axis = op.options.get("axis", None)
            if axis is not None:
                op.options["axis"] = rewrite_axis_for_layout(
                    axis=int(axis),
                    source_layout=original_layout,
                    target_layout=target_layout,
                    rank=rank,
                )
            if len(op.inputs) >= 1:
                axis_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                if axis_tensor is not None:
                    _rewrite_axis_constant_inplace(
                        tensor=axis_tensor,
                        source_layout=original_layout,
                        target_layout=target_layout,
                        rank=rank,
                    )
        elif op_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            if len(op.inputs) >= 2:
                axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if axis_tensor is not None:
                    _rewrite_axis_constant_inplace(
                        tensor=axis_tensor,
                        source_layout=original_layout,
                        target_layout=target_layout,
                        rank=rank,
                    )
        elif op_type in {"SLICE", "STRIDED_SLICE"}:
            for input_name in op.inputs[1:4]:
                vector_tensor = model_ir.tensors.get(str(input_name), None)
                if vector_tensor is not None:
                    _rewrite_vector_constant_inplace(
                        tensor=vector_tensor,
                        perm=logical_layout_permutation(
                            source_layout=original_layout,
                            target_layout=target_layout,
                        ) or [],
                        expected_rank=rank,
                    )
        elif op_type in {"PAD", "PADV2", "MIRROR_PAD"} and len(op.inputs) >= 2:
            pad_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if pad_tensor is not None:
                _rewrite_matrix_constant_inplace(
                    tensor=pad_tensor,
                    perm=logical_layout_permutation(
                        source_layout=original_layout,
                        target_layout=target_layout,
                    ) or [],
                    expected_rank=rank,
                )
        elif op_type == "TRANSPOSE":
            layout_perm = logical_layout_permutation(
                source_layout=original_layout,
                target_layout=target_layout,
            ) or []
            if len(layout_perm) != rank:
                continue
            old_axis_to_new_axis = [0] * rank
            for new_axis, old_axis in enumerate(layout_perm):
                old_axis_to_new_axis[int(old_axis)] = int(new_axis)
            if len(op.inputs) >= 2:
                perm_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if perm_tensor is not None and isinstance(perm_tensor.data, np.ndarray):
                    perm_values = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
                    if len(perm_values) == rank:
                        rewritten_perm = [int(old_axis_to_new_axis[int(axis)]) for axis in perm_values]
                        perm_tensor.data = np.asarray(rewritten_perm, dtype=np.asarray(perm_tensor.data).dtype)
                        perm_tensor.shape = [int(rank)]
                        perm_tensor.shape_signature = [int(rank)]
            elif "perm" in op.options:
                perm_values = [int(v) for v in list(op.options.get("perm", []))]
                if len(perm_values) == rank:
                    op.options["perm"] = [int(old_axis_to_new_axis[int(axis)]) for axis in perm_values]
        elif op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} and len(op.inputs) >= 1:
            output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            if output_shape_tensor is not None:
                _rewrite_vector_constant_inplace(
                    tensor=output_shape_tensor,
                    perm=logical_layout_permutation(
                        source_layout=original_layout,
                        target_layout=target_layout,
                    ) or [],
                    expected_rank=rank,
                )
        elif op_type == "RESHAPE" and len(op.outputs) == 1:
            out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            if out_tensor is not None:
                if len(op.inputs) >= 2:
                    shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                    if shape_tensor is not None and isinstance(shape_tensor.data, np.ndarray):
                        shape_tensor.data = np.asarray(list(out_tensor.shape), dtype=np.asarray(shape_tensor.data).dtype)
                        shape_tensor.shape = [int(len(out_tensor.shape))]
                        shape_tensor.shape_signature = [int(len(out_tensor.shape))]
                op.options["newShape"] = [int(v) for v in list(out_tensor.shape)]


def _rewrite_filter_tensors_for_pytorch(model_ir: ModelIR) -> None:
    rewritten_weights: Set[str] = set()
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type not in {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV", "CONV_3D", "CONV_3D_TRANSPOSE"}:
            continue
        if len(op.inputs) < 2:
            continue
        weight_name = str(op.inputs[1])
        if weight_name in rewritten_weights:
            continue
        tensor = model_ir.tensors.get(weight_name, None)
        if tensor is None or not isinstance(tensor.data, np.ndarray):
            continue
        arr = np.asarray(tensor.data)
        if op_type == "CONV_2D" and arr.ndim == 4:
            tensor.data = np.transpose(arr, (0, 3, 1, 2)).copy()
        elif op_type == "DEPTHWISE_CONV_2D" and arr.ndim == 4:
            permuted = np.transpose(arr, (3, 0, 1, 2)).copy()
            tensor.data = permuted.reshape(int(permuted.shape[0] * permuted.shape[1]), 1, int(permuted.shape[2]), int(permuted.shape[3]))
        elif op_type == "TRANSPOSE_CONV" and arr.ndim == 4:
            tensor.data = np.transpose(arr, (3, 0, 1, 2)).copy()
        elif op_type == "CONV_3D" and arr.ndim == 5:
            tensor.data = np.transpose(arr, (0, 4, 1, 2, 3)).copy()
        elif op_type == "CONV_3D_TRANSPOSE" and arr.ndim == 5:
            tensor.data = np.transpose(arr, (4, 0, 1, 2, 3)).copy()
        else:
            continue
        tensor.shape = [int(v) for v in list(tensor.data.shape)]
        if tensor.shape_signature is not None and len(list(tensor.shape_signature)) == int(arr.ndim):
            tensor.shape_signature = [int(v) for v in list(tensor.shape)]
        rewritten_weights.add(weight_name)


def _synchronize_reshape_targets_with_output_tensors(model_ir: ModelIR) -> None:
    for op in model_ir.operators:
        if str(op.op_type) != "RESHAPE" or len(op.outputs) != 1:
            continue
        out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if out_tensor is None:
            continue
        concrete_shape = [int(v) for v in list(out_tensor.shape)]
        if len(concrete_shape) == 0:
            continue
        op.options["newShape"] = list(concrete_shape)
        if len(op.inputs) < 2:
            continue
        shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        if shape_tensor is None or not isinstance(shape_tensor.data, np.ndarray):
            continue
        dtype = np.asarray(shape_tensor.data).dtype
        shape_tensor.data = np.asarray(concrete_shape, dtype=dtype)
        shape_tensor.shape = [int(len(concrete_shape))]
        shape_tensor.shape_signature = [int(len(concrete_shape))]


def _remove_redundant_layout_transposes(model_ir: ModelIR, original_layouts: Dict[str, str]) -> None:
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    delete_op_indices: Set[int] = set()
    for op_idx, op in enumerate(model_ir.operators):
        if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 1 or len(op.outputs) != 1:
            continue
        input_name = str(op.inputs[0])
        output_name = str(op.outputs[0])
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        reference_tensor = output_tensor if output_tensor is not None else input_tensor
        rank = len(list(reference_tensor.shape)) if reference_tensor is not None else -1
        if rank not in {3, 4, 5}:
            continue
        perm = _read_transpose_perm(model_ir, op)
        input_layout = normalize_logical_layout(original_layouts.get(input_name, LOGICAL_LAYOUT_UNKNOWN))
        output_layout = normalize_logical_layout(original_layouts.get(output_name, LOGICAL_LAYOUT_UNKNOWN))
        remove_as_identity = bool(
            perm is not None
            and (
                perm == list(range(rank))
                or
                (
                    is_channel_last_logical_layout(input_layout)
                    and perm == logical_layout_permutation(
                        source_layout=input_layout,
                        target_layout=channel_first_logical_layout(rank),
                    )
                )
                or (
                    is_channel_last_logical_layout(output_layout)
                    and perm == logical_layout_permutation(
                        source_layout=channel_first_logical_layout(rank),
                        target_layout=output_layout,
                    )
                )
            )
        )
        if not remove_as_identity:
            continue
        if output_name in model_ir.outputs:
            source_tensor = input_tensor if input_tensor is not None else output_tensor
            if source_tensor is not None:
                replacement = _clone_tensor(source_tensor)
                replacement.name = output_name
                model_ir.tensors[output_name] = replacement
            model_ir.operators[int(op_idx)] = OperatorIR(
                op_type="IDENTITY",
                inputs=[input_name],
                outputs=[output_name],
                options={},
            )
            continue
        for consumer_idx in consumers.get(output_name, []):
            consumer = model_ir.operators[int(consumer_idx)]
            consumer.inputs = [input_name if str(v) == output_name else str(v) for v in consumer.inputs]
        delete_op_indices.add(int(op_idx))
        model_ir.tensors.pop(output_name, None)

    if len(delete_op_indices) > 0:
        model_ir.operators = [
            op for op_idx, op in enumerate(model_ir.operators) if int(op_idx) not in delete_op_indices
        ]


def _has_recurrent_sequence_context(model_ir: ModelIR) -> bool:
    recurrent_op_types = {
        "GRU",
        "LSTM",
        "RNN",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if any(str(op.op_type) in recurrent_op_types for op in model_ir.operators):
        return True
    recurrent_name_tokens = ("_gru_", "_lstm_", "_rnn_")
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs) + list(model_ir.tensors.keys()):
        lowered = str(tensor_name).lower()
        if any(token in lowered for token in recurrent_name_tokens):
            return True
    return False


def _repair_orphan_recurrent_step_tensors(model_ir: ModelIR) -> None:
    producer_index: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producer_index[str(output_name)] = int(op_idx)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    for tensor_name in list(model_ir.tensors.keys()):
        tensor_name = str(tensor_name)
        if tensor_name in producer_index or tensor_name in set(str(v) for v in model_ir.inputs):
            continue
        match = re.match(r"^(.+_(?:h|c)_step_)(\d+)$", tensor_name)
        if match is None:
            continue
        shape_tensor_name = f"{match.group(1)}shape_{match.group(2)}"
        replacement_name: Optional[str] = None
        for op in model_ir.operators:
            if str(op.op_type) != "RESHAPE" or len(op.inputs) < 2 or len(op.outputs) != 1:
                continue
            if str(op.inputs[1]) != shape_tensor_name:
                continue
            candidate_name = str(op.outputs[0])
            if candidate_name == tensor_name:
                replacement_name = None
                break
            replacement_name = candidate_name
            break
        if replacement_name is None:
            continue
        for consumer_idx in consumers.get(tensor_name, []):
            consumer = model_ir.operators[int(consumer_idx)]
            consumer.inputs = [
                replacement_name if str(input_name) == tensor_name else str(input_name)
                for input_name in consumer.inputs
            ]
        if tensor_name not in set(str(v) for v in model_ir.outputs):
            model_ir.tensors.pop(tensor_name, None)


def _reject_residual_layout_transposes(model_ir: ModelIR) -> None:
    for op in model_ir.operators:
        if str(op.op_type) != "TRANSPOSE":
            continue
        related_tensor_names = [str(v) for v in list(op.inputs) + list(op.outputs)]
        recurrent_sequence_context = any(
            token in name.lower()
            for name in related_tensor_names
            for token in ("_gru_", "_lstm_", "_rnn_")
        )
        output_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
        output_tensor = model_ir.tensors.get(output_name, None)
        rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
        if rank not in {3, 4, 5}:
            continue
        if recurrent_sequence_context and rank == 3:
            continue
        input_name = str(op.inputs[0]) if len(op.inputs) > 0 else ""
        input_tensor = model_ir.tensors.get(input_name, None)
        input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        output_layout = normalize_logical_layout(output_tensor.logical_layout) if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        if input_layout == LOGICAL_LAYOUT_UNKNOWN and output_layout == LOGICAL_LAYOUT_UNKNOWN:
            continue
        perm = _read_transpose_perm(model_ir, op)
        if perm == _perm_cl_to_cf(rank) or perm == _perm_cf_to_cl(rank):
            raise ModelIRPyTorchExportError(
                "Channel-first normalization failed: residual layout transpose remains. "
                f"op_type={op.op_type} outputs={op.outputs} perm={perm}"
            )


def _align_public_boundary_shapes_to_onnx_contract(model_ir: ModelIR) -> None:
    boundary_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(boundary_map, dict):
        return
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    producer_index: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producer_index[str(output_name)] = int(op_idx)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))
    recurrent_sequence_context = _has_recurrent_sequence_context(model_ir)
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        boundary_shape = boundary_map.get(str(tensor_name), None)
        if tensor is None or not isinstance(boundary_shape, list):
            continue
        if len(boundary_shape) != len(list(tensor.shape)):
            continue
        rank = len(list(tensor.shape))
        preserve_runtime_boundary_shape = False
        if recurrent_sequence_context and rank == 3:
            if str(tensor_name) in set(str(v) for v in model_ir.inputs):
                preserve_runtime_boundary_shape = any(
                    str(model_ir.operators[int(op_idx)].op_type) == "TRANSPOSE"
                    for op_idx in consumers.get(str(tensor_name), [])
                )
            else:
                producer_op_idx = producer_index.get(str(tensor_name), None)
                preserve_runtime_boundary_shape = (
                    producer_op_idx is not None
                    and str(model_ir.operators[int(producer_op_idx)].op_type) in {"TRANSPOSE", "RESHAPE"}
                )
        if preserve_runtime_boundary_shape:
            continue
        tensor.shape_signature = [int(v) for v in list(boundary_shape)]
        tensor.shape = [int(v) if int(v) > 0 else 1 for v in list(boundary_shape)]
        if rank in {3, 4, 5}:
            tensor.logical_layout = normalize_logical_layout(
                public_layout_map.get(
                    str(tensor_name),
                    channel_first_logical_layout(rank),
                )
            )


def validate_channel_first_exportability(model_ir: ModelIR) -> None:
    layout_sensitive_ops = {
        "AVERAGE_POOL_2D",
        "CONCATENATION",
        "CONV_2D",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
        "DEPTHWISE_CONV_2D",
        "DEPTH_TO_SPACE",
        "GATHER",
        "GATHER_ND",
        "MAX_POOL_2D",
        "MEAN",
        "MIRROR_PAD",
        "PAD",
        "PADV2",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SCATTER_ND",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "STRIDED_SLICE",
        "TRANSPOSE_CONV",
        "UNPACK",
    }
    problems: List[str] = []
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type not in layout_sensitive_ops:
            continue
        related_tensor_names = [
            str(v) for v in list(op.inputs) + list(op.outputs)
        ]
        related_lowered_names = [name.lower() for name in related_tensor_names]
        recurrent_sequence_context = any(
            token in name
            for name in related_lowered_names
            for token in ("_gru_", "_lstm_", "_rnn_")
        )
        tensor_names: List[str] = []
        primary_name = _primary_data_input_name(op)
        if primary_name is not None:
            tensor_names.append(str(primary_name))
        tensor_names.extend(str(v) for v in list(op.outputs))
        for tensor_name in tensor_names:
            tensor = model_ir.tensors.get(str(tensor_name), None)
            if tensor is None:
                continue
            rank = len(list(tensor.shape))
            if rank not in {3, 4, 5}:
                continue
            if recurrent_sequence_context and op_type in {"CONCATENATION", "SLICE", "STRIDED_SLICE", "SPLIT"}:
                continue
            layout = normalize_logical_layout(tensor.logical_layout)
            if layout == LOGICAL_LAYOUT_UNKNOWN or is_channel_last_logical_layout(layout):
                problems.append(
                    f"op_type={op_type} tensor={tensor_name} logical_layout={layout}"
                )
    if len(problems) > 0:
        raise ModelIRPyTorchExportError(
            "Channel-first normalization failed: semantic layout annotations are incomplete. "
            f"problems={sorted(set(problems))}"
        )


def normalize_model_ir_for_pytorch_channel_first(model_ir: ModelIR) -> ModelIR:
    normalized = copy.deepcopy(model_ir)
    infer_model_ir_logical_layouts(normalized)
    annotation_problems = validate_model_ir_layout_annotations(normalized)
    if len(annotation_problems) > 0:
        raise ModelIRPyTorchExportError(
            "Channel-first normalization failed: invalid semantic layout annotations. "
            f"problems={annotation_problems}"
        )
    original_layouts = {
        str(name): normalize_logical_layout(tensor.logical_layout)
        for name, tensor in normalized.tensors.items()
    }
    _rewrite_layout_sensitive_ops(normalized, original_layouts)
    kernel_weight_tensor_names = _collect_kernel_weight_tensor_names(normalized)
    for tensor_name, tensor in normalized.tensors.items():
        if str(tensor_name) in kernel_weight_tensor_names:
            continue
        _permute_tensor_to_channel_first_inplace(tensor)
    _synchronize_reshape_targets_with_output_tensors(normalized)
    _rewrite_filter_tensors_for_pytorch(normalized)
    _remove_redundant_layout_transposes(normalized, original_layouts)
    _repair_orphan_recurrent_step_tensors(normalized)
    _align_public_boundary_shapes_to_onnx_contract(normalized)
    normalized.metadata["assume_channel_last_layout_tensor_names"] = []
    _reject_residual_layout_transposes(normalized)
    validate_channel_first_exportability(normalized)
    return normalized


def _collect_model_op_types(model_ir: ModelIR) -> Set[str]:
    ops: Set[str] = set()
    for op in model_ir.operators:
        ops.add(str(op.op_type))
    for subgraph in model_ir.subgraphs:
        ops.update(_collect_model_op_types(subgraph))
    return ops


def _ensure_supported_ops(model_ir: ModelIR) -> None:
    unsupported = sorted(
        {
            op_type
            for op_type in _collect_model_op_types(model_ir)
            if op_type not in SUPPORTED_TORCH_KERNEL_OP_TYPES and op_type not in {"MODEL"}
        }
    )
    if len(unsupported) > 0:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support some op types in this model. "
            f"unsupported_op_types={unsupported}"
        )


def _ensure_no_custom_ops(model_ir: ModelIR) -> None:
    custom_ops = sorted({str(op.op_type) for op in model_ir.operators if str(op.op_type) == "CUSTOM"})
    if len(custom_ops) > 0:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support CUSTOM ops."
        )


def _serializable_tensor_meta(tensor: TensorIR) -> Dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(v) for v in list(tensor.shape)],
        "shape_signature": (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else None
        ),
        "is_variable": bool(tensor.is_variable),
        "has_data": bool(isinstance(tensor.data, np.ndarray)),
        "logical_layout": normalize_logical_layout(tensor.logical_layout),
    }


def _serializable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serializable_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_serializable_value(v) for v in value]
    if isinstance(value, list):
        return [_serializable_value(v) for v in value]
    return value


def _build_metadata_payload(model_ir: ModelIR) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": str(model_ir.name),
        "description": str(model_ir.description),
        "inputs": [str(v) for v in model_ir.inputs],
        "outputs": [str(v) for v in model_ir.outputs],
        "tensors": {
            str(name): _serializable_tensor_meta(tensor)
            for name, tensor in model_ir.tensors.items()
        },
        "operators": [
            {
                "op_type": str(op.op_type),
                "inputs": [str(v) for v in op.inputs],
                "outputs": [str(v) for v in op.outputs],
                "options": _serializable_value(dict(op.options)),
                "axis_semantics": _serializable_value(dict(op.axis_semantics)),
                "version": int(op.version),
            }
            for op in model_ir.operators
        ],
        "public_layouts": _serializable_value(dict(model_ir.metadata.get("onnx_public_layout_map", {}))),
    }


def _build_tflite_backed_metadata_payload(
    *,
    model_ir: ModelIR,
    tflite_file_name: str,
) -> Dict[str, Any]:
    inferred = copy.deepcopy(model_ir)
    infer_model_ir_logical_layouts(inferred)
    boundary_shape_map = inferred.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}

    public_tensor_names = {
        str(name) for name in list(inferred.inputs) + list(inferred.outputs)
    }
    tensors: Dict[str, Dict[str, Any]] = {}
    for tensor_name in sorted(public_tensor_names):
        tensor = inferred.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        tensor_meta = _serializable_tensor_meta(tensor)
        boundary_shape = boundary_shape_map.get(str(tensor_name), None)
        if isinstance(boundary_shape, list) and len(boundary_shape) == len(tensor_meta["shape"]):
            tensor_meta["shape"] = [max(1, int(v)) if int(v) >= 0 else 1 for v in boundary_shape]
            tensor_meta["shape_signature"] = [int(v) for v in boundary_shape]
        tensor_meta["has_data"] = False
        tensors[str(tensor_name)] = tensor_meta

    current_public_layouts = {
        str(name): normalize_logical_layout(inferred.tensors[str(name)].logical_layout)
        for name in public_tensor_names
        if str(name) in inferred.tensors
    }
    return {
        "schema_version": 1,
        "execution_backend": "tflite",
        "name": str(inferred.name),
        "description": str(inferred.description),
        "inputs": [str(v) for v in inferred.inputs],
        "outputs": [str(v) for v in inferred.outputs],
        "tensors": tensors,
        "operators": [],
        "public_layouts": _serializable_value(
            dict(inferred.metadata.get("onnx_public_layout_map", {}))
        ),
        "current_public_layouts": _serializable_value(current_public_layouts),
        "boundary_shape_signatures": _serializable_value(boundary_shape_map),
        "tflite_file_name": str(tflite_file_name),
    }


def _build_saved_model_backed_metadata_payload(
    *,
    model_ir: ModelIR,
    saved_model_dir_name: str,
) -> Dict[str, Any]:
    metadata = _build_tflite_backed_metadata_payload(
        model_ir=model_ir,
        tflite_file_name="",
    )
    metadata["execution_backend"] = "saved_model"
    metadata["saved_model_dir_name"] = str(saved_model_dir_name)
    metadata.pop("tflite_file_name", None)
    return metadata


def _extract_string_normalizer_config_from_onnx_graph(
    onnx_graph: Any,
) -> Optional[Dict[str, Any]]:
    if onnx_graph is None:
        return None
    graph = getattr(onnx_graph, "graph", None)
    if graph is None or len(list(graph.node)) != 1:
        return None
    node = list(graph.node)[0]
    if str(node.op_type) != "StringNormalizer":
        return None
    if len(list(graph.input)) == 0 or len(list(graph.output)) == 0:
        return None
    attributes = {
        str(attr.name): onnx.helper.get_attribute_value(attr)
        for attr in node.attribute
    }
    stopwords = []
    for value in list(attributes.get("stopwords", [])):
        if isinstance(value, bytes):
            stopwords.append(value.decode("utf-8"))
        else:
            stopwords.append(str(value))
    case_change_action = attributes.get("case_change_action", b"")
    locale = attributes.get("locale", b"")
    return {
        "input_name": str(node.input[0]),
        "output_name": str(node.output[0]),
        "case_change_action": (
            case_change_action.decode("utf-8")
            if isinstance(case_change_action, bytes)
            else str(case_change_action)
        ),
        "is_case_sensitive": bool(int(attributes.get("is_case_sensitive", 1))),
        "locale": locale.decode("utf-8") if isinstance(locale, bytes) else str(locale),
        "stopwords": stopwords,
    }


def export_pytorch_package_from_string_normalizer_onnx(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    onnx_graph: Any,
) -> str:
    config = _extract_string_normalizer_config_from_onnx_graph(onnx_graph)
    if config is None:
        raise ModelIRPyTorchExportError(
            "StringNormalizer fallback requires a single-op StringNormalizer ONNX graph."
        )
    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)
    metadata = {
        "schema_version": 1,
        "execution_backend": "string_normalizer",
        "name": str(model_ir.name),
        "description": str(model_ir.description),
        "inputs": [str(v) for v in model_ir.inputs],
        "outputs": [str(v) for v in model_ir.outputs],
        "tensors": {
            str(name): _serializable_tensor_meta(model_ir.tensors[str(name)])
            for name in list(model_ir.inputs) + list(model_ir.outputs)
            if str(name) in model_ir.tensors
        },
        "operators": [],
        "public_layouts": {},
        "string_normalizer": config,
    }
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)


def _make_tensor_storage_name_map(model_ir: ModelIR) -> Dict[str, str]:
    used_names: Set[str] = set()
    storage_name_map: Dict[str, str] = {}
    for tensor_name, tensor in sorted(model_ir.tensors.items()):
        if not isinstance(tensor.data, np.ndarray):
            continue
        base_name = re.sub(r"[^0-9A-Za-z_]", "_", str(tensor_name)).strip("_")
        if base_name == "":
            base_name = "tensor"
        if base_name[0].isdigit():
            base_name = f"tensor_{base_name}"
        candidate = base_name
        suffix = 1
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        storage_name_map[str(tensor_name)] = candidate
    return storage_name_map


_DIRECT_CODEGEN_MODULE_OP_TYPES: Set[str] = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "TRANSPOSE_CONV",
    "CONV_3D",
    "CONV_3D_TRANSPOSE",
}

_DIRECT_CODEGEN_UNARY_EXPRESSIONS: Dict[str, str] = {
    "ABS": "torch.abs({x})",
    "ACOS": "torch.acos({x})",
    "ASIN": "torch.asin({x})",
    "ATAN": "torch.atan({x})",
    "CEIL": "torch.ceil({x})",
    "COS": "torch.cos({x})",
    "ELU": "F.elu({x})",
    "EXP": "torch.exp({x})",
    "FLOOR": "torch.floor({x})",
    "HARD_SWISH": "F.hardswish({x})",
    "IDENTITY": "{x}",
    "LEAKY_RELU": "F.leaky_relu({x}, negative_slope={alpha})",
    "LOG": "torch.log({x})",
    "LOGICAL_NOT": "torch.logical_not({x})",
    "NEG": "torch.neg({x})",
    "RELU": "torch.relu({x})",
    "RELU6": "torch.clamp({x}, min=0.0, max=6.0)",
    "ROUND": "torch.round({x})",
    "RSQRT": "torch.rsqrt({x})",
    "SIGMOID": "torch.sigmoid({x})",
    "SIGN": "torch.sign({x})",
    "SIN": "torch.sin({x})",
    "SQRT": "torch.sqrt({x})",
    "SQUARE": "torch.square({x})",
    "TAN": "torch.tan({x})",
    "TANH": "torch.tanh({x})",
}

_DIRECT_CODEGEN_BINARY_FUNCTIONS: Dict[str, str] = {
    "ADD": "torch.add",
    "DIV": "torch.div",
    "MAXIMUM": "torch.maximum",
    "MINIMUM": "torch.minimum",
    "MUL": "torch.mul",
    "POW": "torch.pow",
    "SUB": "torch.sub",
}

_DIRECT_CODEGEN_SUPPORTED_OP_TYPES: Set[str] = (
    set(_DIRECT_CODEGEN_MODULE_OP_TYPES)
    | set(_DIRECT_CODEGEN_UNARY_EXPRESSIONS.keys())
    | set(_DIRECT_CODEGEN_BINARY_FUNCTIONS.keys())
    | {
        "BATCH_MATMUL",
        "CAST",
        "CONCATENATION",
        "EXPAND_DIMS",
        "FILL",
        "MEAN",
        "MIRROR_PAD",
        "PACK",
        "PAD",
        "PADV2",
        "RANGE",
        "REDUCE_ANY",
        "REDUCE_MAX",
        "REDUCE_MIN",
        "REDUCE_PROD",
        "RESHAPE",
        "SHAPE",
        "SOFTMAX",
        "SPLIT",
        "SQUEEZE",
        "SUM",
        "TILE",
        "TRANSPOSE",
        "UNPACK",
        "WHERE",
    }
)


def _sanitize_python_identifier(name: str, *, prefix: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", str(name)).strip("_")
    if identifier == "":
        identifier = str(prefix)
    if identifier[0].isdigit():
        identifier = f"{prefix}_{identifier}"
    if keyword.iskeyword(identifier):
        identifier = f"{identifier}_{prefix}"
    return identifier


def _make_unique_identifier(base_name: str, used_names: Set[str]) -> str:
    candidate = str(base_name)
    suffix = 1
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _build_tensor_var_name_map(model_ir: ModelIR) -> Dict[str, str]:
    used_names: Set[str] = set()
    mapping: Dict[str, str] = {}
    for tensor_name in list(model_ir.inputs) + [str(out) for op in model_ir.operators for out in op.outputs]:
        if str(tensor_name) in mapping:
            continue
        base = _sanitize_python_identifier(str(tensor_name), prefix="t")
        mapping[str(tensor_name)] = _make_unique_identifier(base, used_names)
    return mapping


def _build_buffer_attr_name_map(
    *,
    model_ir: ModelIR,
    tensor_storage_name_map: Dict[str, str],
    excluded_tensor_names: Set[str],
) -> Dict[str, str]:
    used_names: Set[str] = set()
    mapping: Dict[str, str] = {}
    for tensor_name, tensor in sorted(model_ir.tensors.items()):
        if str(tensor_name) in excluded_tensor_names or not isinstance(tensor.data, np.ndarray):
            continue
        storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
        base = _sanitize_python_identifier(f"const_{storage_name}", prefix="const")
        mapping[str(tensor_name)] = _make_unique_identifier(base, used_names)
    return mapping


def _direct_codegen_module_attr_name(op_index: int, op_type: str) -> str:
    base = _sanitize_python_identifier(f"op_{op_index}_{str(op_type).lower()}", prefix="op")
    return base


def _ensure_direct_codegen_supported(model_ir: ModelIR) -> None:
    unsupported = sorted(
        {str(op.op_type) for op in model_ir.operators if str(op.op_type) not in _DIRECT_CODEGEN_SUPPORTED_OP_TYPES}
    )
    if len(unsupported) > 0:
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen does not support some op types in this model. "
            f"unsupported_op_types={unsupported}"
        )


def _is_direct_codegen_unsupported_error(ex: BaseException) -> bool:
    return "Native PyTorch-like model.py codegen does not support some op types in this model." in str(ex)


def _write_generated_package_common_files(output_folder_path: str) -> None:
    package_dir = Path(output_folder_path)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "from .model import Model, load_model\n",
        encoding="utf-8",
    )
    (package_dir / "runtime.py").write_text(
        "from onnx2tf.tflite_builder.pytorch_package_runtime import load_generated_model_package\n",
        encoding="utf-8",
    )


def _write_wrapper_model_file(output_folder_path: str) -> None:
    package_dir = Path(output_folder_path)
    (package_dir / "model.py").write_text(
        "from __future__ import annotations\n\n"
        "from typing import Any, Callable, cast\n\n"
        "from pathlib import Path\n\n"
        "import torch\n\n"
        "from .runtime import load_generated_model_package\n\n"
        "PACKAGE_DIR = Path(__file__).resolve().parent\n\n"
        "class Model(torch.nn.Module):\n"
        "    def __init__(self, device: str | None = None, eval_mode: bool = True):\n"
        "        super().__init__()\n"
        "        self._model: Any = load_generated_model_package(\n"
        "            package_dir=str(PACKAGE_DIR),\n"
        "            device=device,\n"
        "            eval_mode=eval_mode,\n"
        "        )\n\n"
        "    def forward(self, *args: Any, **kwargs: Any) -> Any:\n"
        "        return self._model(*args, **kwargs)\n\n"
        "    def forward_named(self, *args: Any, **kwargs: Any) -> Any:\n"
        "        forward_named = getattr(self._model, 'forward_named', None)\n"
        "        if callable(forward_named):\n"
        "            return cast(Callable[..., Any], forward_named)(*args, **kwargs)\n"
        "        return self.forward(*args, **kwargs)\n\n"
        "def load_model(device: str | None = None, eval_mode: bool = True) -> Model:\n"
        "    return Model(device=device, eval_mode=eval_mode)\n",
        encoding="utf-8",
    )


_RUNTIME_SUPPORTED_CUSTOM_CODES: Set[str] = {
    "ONNX_SLICE",
}


def _supports_runtime_wrapper_model_ir(model_ir: ModelIR) -> bool:
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type in SUPPORTED_TORCH_KERNEL_OP_TYPES:
            continue
        if op_type == "CUSTOM":
            custom_code = str(op.options.get("customCode", "")).upper()
            if custom_code in _RUNTIME_SUPPORTED_CUSTOM_CODES:
                continue
        return False
    return True


def _export_runtime_wrapper_package_from_model_ir(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
) -> str:
    if not _supports_runtime_wrapper_model_ir(model_ir):
        raise ModelIRPyTorchExportError(
            "PyTorch runtime wrapper export does not support some op types in this model."
        )
    os.makedirs(output_folder_path, exist_ok=True)
    tensor_storage_name_map = _make_tensor_storage_name_map(model_ir)
    metadata = _build_metadata_payload(model_ir)
    metadata["tensor_storage_names"] = dict(tensor_storage_name_map)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    import torch
    state_dict: Dict[str, Any] = {}
    for tensor_name, tensor in model_ir.tensors.items():
        if not isinstance(tensor.data, np.ndarray):
            continue
        storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
        state_dict[storage_name] = torch.as_tensor(np.asarray(tensor.data))
    torch.save(state_dict, os.path.join(output_folder_path, "state_dict.pth"))
    return str(output_folder_path)


def _write_native_model_file(
    output_folder_path: str,
    *,
    model_ir: ModelIR,
    metadata: Dict[str, Any],
    tensor_storage_name_map: Dict[str, str],
) -> None:
    package_dir = Path(output_folder_path)
    _ensure_direct_codegen_supported(model_ir)
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    metadata_literal = repr(metadata_json)
    tensor_var_names = _build_tensor_var_name_map(model_ir)
    module_param_tensor_names: Set[str] = set()
    module_init_lines: List[str] = []
    weight_load_lines: List[str] = []
    op_module_attr_names: Dict[int, str] = {}

    def _shape_literal(values: Sequence[int]) -> str:
        return repr(tuple(int(v) for v in list(values)))

    def _target_shape_literal(tensor_name: str) -> str:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            return "None"
        return repr([int(v) for v in list(tensor.shape)])

    for op_index, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        if op_type not in _DIRECT_CODEGEN_MODULE_OP_TYPES:
            continue
        attr_name = _direct_codegen_module_attr_name(op_index, op_type)
        op_module_attr_names[int(op_index)] = attr_name
        weight_name = str(op.inputs[1]) if len(op.inputs) >= 2 else ""
        bias_name = ""
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D"} and len(op.inputs) >= 3 and str(op.inputs[2]) != "":
            bias_name = str(op.inputs[2])
        elif op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} and len(op.inputs) >= 4 and str(op.inputs[3]) != "":
            bias_name = str(op.inputs[3])
        module_param_tensor_names.add(weight_name)
        if bias_name != "":
            module_param_tensor_names.add(bias_name)
        weight_tensor = model_ir.tensors[weight_name]
        input_tensor_name = str(op.inputs[0]) if op_type not in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} else str(op.inputs[2])
        input_tensor = model_ir.tensors[input_tensor_name]
        options = dict(op.options)
        if op_type == "CONV_2D":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.Conv2d(",
                    f"    in_channels={int(input_tensor.shape[1])},",
                    f"    out_channels={int(weight_tensor.shape[0])},",
                    f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                    f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(weight_tensor.shape[2]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[3]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                    f"    groups={max(1, int(input_tensor.shape[1]) // max(1, int(weight_tensor.shape[1])))},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "DEPTHWISE_CONV_2D":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.Conv2d(",
                    f"    in_channels={int(input_tensor.shape[1])},",
                    f"    out_channels={int(weight_tensor.shape[0])},",
                    f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                    f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(weight_tensor.shape[2]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[3]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                    f"    groups={int(input_tensor.shape[1])},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "TRANSPOSE_CONV":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.ConvTranspose2d(",
                    f"    in_channels={int(weight_tensor.shape[0])},",
                    f"    out_channels={int(weight_tensor.shape[1])},",
                    f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                    f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(weight_tensor.shape[2]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[3]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "CONV_3D":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.Conv3d(",
                    f"    in_channels={int(input_tensor.shape[1])},",
                    f"    out_channels={int(weight_tensor.shape[0])},",
                    f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                    f"    stride={_shape_literal([int(options.get('strideD', 1)), int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(weight_tensor.shape[2]) - 1) * int(options.get('dilationDFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[3]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[4]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    dilation={_shape_literal([int(options.get('dilationDFactor', 1)), int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                    f"    groups={max(1, int(input_tensor.shape[1]) // max(1, int(weight_tensor.shape[1])))},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "CONV_3D_TRANSPOSE":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.ConvTranspose3d(",
                    f"    in_channels={int(weight_tensor.shape[0])},",
                    f"    out_channels={int(weight_tensor.shape[1])},",
                    f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                    f"    stride={_shape_literal([int(options.get('strideD', 1)), int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(weight_tensor.shape[2]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[3]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[4]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        weight_load_lines.append(
            f"_copy_tensor_data(self.{attr_name}.weight, _lookup_state_tensor(raw_state_dict, {weight_name!r}, self.tensor_storage_names))"
        )
        if bias_name != "":
            weight_load_lines.append(
                f"_copy_tensor_data(self.{attr_name}.bias, _lookup_state_tensor(raw_state_dict, {bias_name!r}, self.tensor_storage_names))"
            )

    buffer_attr_names = _build_buffer_attr_name_map(
        model_ir=model_ir,
        tensor_storage_name_map=tensor_storage_name_map,
        excluded_tensor_names=module_param_tensor_names,
    )
    buffer_init_lines: List[str] = []
    buffer_load_lines: List[str] = []
    for tensor_name, attr_name in buffer_attr_names.items():
        tensor = model_ir.tensors[str(tensor_name)]
        dtype_name = str(tensor.dtype).upper()
        shape_values = [int(v) for v in list(tensor.shape)]
        if bool(tensor.is_variable):
            buffer_init_lines.append(
                f"self.register_parameter({attr_name!r}, torch.nn.Parameter(torch.zeros({repr(shape_values)}, dtype=_torch_dtype({dtype_name!r})), requires_grad=False))"
            )
        else:
            buffer_init_lines.append(
                f"self.register_buffer({attr_name!r}, torch.zeros({repr(shape_values)}, dtype=_torch_dtype({dtype_name!r})), persistent=True)"
            )
        buffer_load_lines.append(
            f"_copy_tensor_data(getattr(self, {attr_name!r}), _lookup_state_tensor(raw_state_dict, {tensor_name!r}, self.tensor_storage_names))"
        )

    def _tensor_expr(tensor_name: str) -> str:
        if str(tensor_name) in tensor_var_names:
            return str(tensor_var_names[str(tensor_name)])
        if str(tensor_name) in buffer_attr_names:
            return f"self.{buffer_attr_names[str(tensor_name)]}"
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen could not resolve a tensor expression. "
            f"tensor={tensor_name}"
        )

    forward_lines: List[str] = []
    for op_index, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        outputs = [str(v) for v in op.outputs]
        output_vars = [tensor_var_names[str(name)] for name in outputs]
        output_target_shape = _target_shape_literal(outputs[0]) if len(outputs) == 1 else "None"
        if op_type in _DIRECT_CODEGEN_MODULE_OP_TYPES:
            attr_name = op_module_attr_names[int(op_index)]
            fused = str(op.options.get("fusedActivationFunction", "NONE"))
            if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                forward_lines.append(
                    f"{output_vars[0]} = _apply_module_conv2d(self.{attr_name}, {_tensor_expr(str(op.inputs[0]))}, target_shape={output_target_shape}, fused={fused!r})"
                )
            elif op_type == "TRANSPOSE_CONV":
                output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                fallback_shape = (
                    [int(v) for v in np.asarray(output_shape_tensor.data).reshape(-1).tolist()]
                    if output_shape_tensor is not None and isinstance(output_shape_tensor.data, np.ndarray)
                    else [int(v) for v in list(model_ir.tensors[outputs[0]].shape)]
                )
                forward_lines.append(
                    f"{output_vars[0]} = _apply_module_transpose_conv2d(self.{attr_name}, {_tensor_expr(str(op.inputs[2]))}, target_shape={output_target_shape}, fallback_shape={repr(fallback_shape)}, fused={fused!r})"
                )
            elif op_type == "CONV_3D":
                forward_lines.append(
                    f"{output_vars[0]} = _apply_module_conv3d(self.{attr_name}, {_tensor_expr(str(op.inputs[0]))}, target_shape={output_target_shape}, fused={fused!r})"
                )
            elif op_type == "CONV_3D_TRANSPOSE":
                output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                fallback_shape = (
                    [int(v) for v in np.asarray(output_shape_tensor.data).reshape(-1).tolist()]
                    if output_shape_tensor is not None and isinstance(output_shape_tensor.data, np.ndarray)
                    else [int(v) for v in list(model_ir.tensors[outputs[0]].shape)]
                )
                forward_lines.append(
                    f"{output_vars[0]} = _apply_module_transpose_conv3d(self.{attr_name}, {_tensor_expr(str(op.inputs[2]))}, target_shape={output_target_shape}, fallback_shape={repr(fallback_shape)}, fused={fused!r})"
                )
            continue
        if op_type in _DIRECT_CODEGEN_BINARY_FUNCTIONS:
            fn_name = _DIRECT_CODEGEN_BINARY_FUNCTIONS[op_type]
            fused = str(op.options.get("fusedActivationFunction", "NONE"))
            forward_lines.append(
                f"{output_vars[0]} = _apply_binary({fn_name}, {_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, target_shape={output_target_shape}, fused={fused!r})"
            )
            continue
        if op_type in _DIRECT_CODEGEN_UNARY_EXPRESSIONS:
            template = _DIRECT_CODEGEN_UNARY_EXPRESSIONS[op_type]
            if op_type == "LEAKY_RELU":
                expr = template.format(x=_tensor_expr(str(op.inputs[0])), alpha=float(op.options.get("alpha", 0.2)))
            else:
                expr = template.format(x=_tensor_expr(str(op.inputs[0])))
            forward_lines.append(
                f"{output_vars[0]} = _align_tensor_to_target_shape({expr}, {output_target_shape})"
            )
            continue
        if op_type == "CAST":
            out_dtype = str(op.options.get("outDataType", "FLOAT32"))
            forward_lines.append(
                f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}.to(dtype=_torch_dtype({out_dtype!r}))"
            )
            continue
        if op_type == "RESHAPE":
            if len(op.inputs) >= 2:
                shape_expr = f"_shape_list({_tensor_expr(str(op.inputs[1]))})"
            else:
                shape_expr = repr([int(v) for v in list(op.options.get('newShape', []))])
            forward_lines.append(
                f"{output_vars[0]} = torch.reshape({_tensor_expr(str(op.inputs[0]))}, [int(v) for v in {shape_expr}])"
            )
            continue
        if op_type == "TRANSPOSE":
            if len(op.inputs) >= 2:
                perm_expr = f"_shape_list({_tensor_expr(str(op.inputs[1]))})"
            else:
                perm_expr = repr([int(v) for v in list(op.options.get('perm', []))])
            forward_lines.append(
                f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}.permute(*[int(v) for v in {perm_expr}]).contiguous()"
            )
            continue
        if op_type == "EXPAND_DIMS":
            axis_expr = (
                f"_coerce_scalar_axis({_tensor_expr(str(op.inputs[1]))}, device={_tensor_expr(str(op.inputs[0]))}.device)"
                if len(op.inputs) >= 2
                else repr(int(op.options.get("axis", 0)))
            )
            forward_lines.append(
                f"{output_vars[0]} = torch.unsqueeze({_tensor_expr(str(op.inputs[0]))}, dim={axis_expr})"
            )
            continue
        if op_type == "SQUEEZE":
            squeeze_dims = [int(v) for v in list(op.options.get("squeezeDims", []))]
            if len(squeeze_dims) == 0:
                forward_lines.append(f"{output_vars[0]} = torch.squeeze({_tensor_expr(str(op.inputs[0]))})")
            else:
                forward_lines.append(f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}")
                for axis in sorted(squeeze_dims, reverse=True):
                    forward_lines.append(
                        f"{output_vars[0]} = torch.squeeze({output_vars[0]}, dim=_normalize_dim({int(axis)}, {output_vars[0]}.ndim))"
                    )
            continue
        if op_type == "CONCATENATION":
            axis = int(op.options.get("axis", 0))
            inputs_expr = ", ".join(_tensor_expr(str(name)) for name in op.inputs)
            forward_lines.append(
                f"{output_vars[0]} = _apply_concat([{inputs_expr}], axis={axis}, target_shape={output_target_shape}, fused={str(op.options.get('fusedActivationFunction', 'NONE'))!r})"
            )
            continue
        if op_type == "PACK":
            axis = int(op.options.get("axis", 0))
            inputs_expr = ", ".join(_tensor_expr(str(name)) for name in op.inputs)
            forward_lines.append(
                f"{output_vars[0]} = torch.stack([{inputs_expr}], dim={axis})"
            )
            continue
        if op_type == "UNPACK":
            axis = int(op.options.get("axis", 0))
            forward_lines.append(
                f"{', '.join(output_vars)} = list(torch.unbind({_tensor_expr(str(op.inputs[0]))}, dim=_normalize_dim({axis}, {_tensor_expr(str(op.inputs[0]))}.ndim)))"
            )
            continue
        if op_type == "SPLIT":
            data_expr = _tensor_expr(str(op.inputs[-1]))
            if len(op.inputs) >= 2:
                axis_expr = (
                    f"_coerce_scalar_axis({_tensor_expr(str(op.inputs[0]))}, device={data_expr}.device)"
                )
            else:
                axis_expr = repr(int(op.options.get("axis", 0)))
            sections = int(op.options.get("numSplits", len(outputs)))
            forward_lines.append(
                f"{', '.join(output_vars)} = list(torch.tensor_split({data_expr}, {sections}, dim=_normalize_dim({axis_expr}, {data_expr}.ndim)))"
            )
            continue
        if op_type == "SHAPE":
            out_dtype = str(op.options.get("outType", "INT32"))
            forward_lines.append(
                f"{output_vars[0]} = torch.tensor(list({_tensor_expr(str(op.inputs[0]))}.shape), dtype=_torch_dtype({out_dtype!r}), device={_tensor_expr(str(op.inputs[0]))}.device)"
            )
            continue
        if op_type == "FILL":
            forward_lines.append(
                f"{output_vars[0]} = torch.full([int(v) for v in _shape_list({_tensor_expr(str(op.inputs[0]))})], {_tensor_expr(str(op.inputs[1]))}.reshape(-1)[0].item(), dtype={_tensor_expr(str(op.inputs[1]))}.dtype, device={_tensor_expr(str(op.inputs[1]))}.device)"
            )
            continue
        if op_type == "RANGE":
            start_expr = _tensor_expr(str(op.inputs[0]))
            limit_expr = _tensor_expr(str(op.inputs[1]))
            delta_expr = _tensor_expr(str(op.inputs[2]))
            forward_lines.append(
                f"{output_vars[0]} = torch.arange(start={start_expr}.reshape(-1)[0].item(), end={limit_expr}.reshape(-1)[0].item(), step={delta_expr}.reshape(-1)[0].item(), device={start_expr}.device, dtype={start_expr}.dtype)"
            )
            continue
        if op_type == "SOFTMAX":
            axis = op.options.get("axis", None)
            axis_expr = repr(int(axis)) if axis is not None else "None"
            beta = float(op.options.get("beta", 1.0))
            forward_lines.append(
                f"{output_vars[0]} = _apply_softmax({_tensor_expr(str(op.inputs[0]))}, axis={axis_expr}, beta={beta}, target_shape={output_target_shape})"
            )
            continue
        if op_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            reducer_map = {
                "SUM": "_reduce_sum",
                "MEAN": "_reduce_mean",
                "REDUCE_MAX": "_reduce_max",
                "REDUCE_MIN": "_reduce_min",
                "REDUCE_PROD": "_reduce_prod",
                "REDUCE_ANY": "_reduce_any",
            }
            axis_expr = (
                f"_normalize_axes({_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[0]))}.ndim)"
                if len(op.inputs) >= 2
                else "None"
            )
            keepdims = bool(op.options.get("keepDims", True))
            forward_lines.append(
                f"{output_vars[0]} = _align_tensor_to_target_shape({reducer_map[op_type]}({_tensor_expr(str(op.inputs[0]))}, {axis_expr}, {keepdims}), {output_target_shape})"
            )
            continue
        if op_type == "PAD":
            forward_lines.append(
                f"{output_vars[0]} = F.pad({_tensor_expr(str(op.inputs[0]))}, _to_torch_pad_arg({_tensor_expr(str(op.inputs[1]))}), mode='constant', value=0.0)"
            )
            continue
        if op_type == "PADV2":
            forward_lines.append(
                f"{output_vars[0]} = F.pad({_tensor_expr(str(op.inputs[0]))}, _to_torch_pad_arg({_tensor_expr(str(op.inputs[1]))}), mode='constant', value=float({_tensor_expr(str(op.inputs[2]))}.reshape(-1)[0].item()))"
            )
            continue
        if op_type == "MIRROR_PAD":
            forward_lines.append(
                f"{output_vars[0]} = F.pad({_tensor_expr(str(op.inputs[0]))}, _to_torch_pad_arg({_tensor_expr(str(op.inputs[1]))}), mode='reflect')"
            )
            continue
        if op_type == "WHERE":
            if len(op.inputs) == 1:
                forward_lines.append(
                    f"{output_vars[0]} = torch.nonzero({_tensor_expr(str(op.inputs[0]))}, as_tuple=False)"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = torch.where({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))})"
                )
            continue
        if op_type == "TILE":
            forward_lines.append(
                f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}.repeat(*[int(v) for v in _shape_list({_tensor_expr(str(op.inputs[1]))})])"
            )
            continue
        if op_type == "BATCH_MATMUL":
            x_expr = _tensor_expr(str(op.inputs[0]))
            y_expr = _tensor_expr(str(op.inputs[1]))
            adj_x = bool(op.options.get("adjX", False))
            adj_y = bool(op.options.get("adjY", False))
            forward_lines.append(f"_tmp_x_{op_index} = {x_expr}")
            forward_lines.append(f"_tmp_y_{op_index} = {y_expr}")
            if adj_x:
                forward_lines.append(f"_tmp_x_{op_index} = _tmp_x_{op_index}.transpose(-1, -2)")
            if adj_y:
                forward_lines.append(f"_tmp_y_{op_index} = _tmp_y_{op_index}.transpose(-1, -2)")
            forward_lines.append(
                f"{output_vars[0]} = _align_tensor_to_target_shape(torch.matmul(_tmp_x_{op_index}, _tmp_y_{op_index}), {output_target_shape})"
            )
            continue
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen hit an unimplemented op emitter. "
            f"op_type={op_type}"
        )

    helper_source = (
        "def _normalize_tensor_name(name: str) -> str:\n"
        "    normalized = str(name).split(\":\")[0]\n"
        "    if normalized.startswith(\"serving_default_\"):\n"
        "        normalized = normalized[len(\"serving_default_\") :]\n"
        "    return normalized\n\n"
        "_TORCH_DTYPE_BY_TFLITE_DTYPE: Dict[str, torch.dtype] = {\n"
        "    'BOOL': torch.bool,\n"
        "    'INT8': torch.int8,\n"
        "    'INT16': torch.int16,\n"
        "    'INT32': torch.int32,\n"
        "    'INT64': torch.int64,\n"
        "    'UINT8': torch.uint8,\n"
        "    'FLOAT16': torch.float16,\n"
        "    'FLOAT32': torch.float32,\n"
        "    'FLOAT64': torch.float64,\n"
        "}\n\n"
        "def _torch_dtype(dtype_name: str) -> torch.dtype:\n"
        "    key = str(dtype_name).upper()\n"
        "    if key not in _TORCH_DTYPE_BY_TFLITE_DTYPE:\n"
        "        raise RuntimeError(f'Unsupported dtype for PyTorch runtime: {dtype_name}')\n"
        "    return _TORCH_DTYPE_BY_TFLITE_DTYPE[key]\n\n"
        "def _default_tensor_storage_name(tensor_name: str) -> str:\n"
        "    base_name = re.sub(r'[^0-9A-Za-z_]', '_', str(tensor_name)).strip('_')\n"
        "    if base_name == '':\n"
        "        base_name = 'tensor'\n"
        "    if base_name[0].isdigit():\n"
        "        base_name = f'tensor_{base_name}'\n"
        "    return base_name\n\n"
        "def _resolve_named_input_value(kwargs: Dict[str, Any], expected_name: str) -> Any:\n"
        "    if str(expected_name) in kwargs:\n"
        "        return kwargs[str(expected_name)]\n"
        "    normalized_expected_name = _normalize_tensor_name(str(expected_name))\n"
        "    canonical_expected_name = re.sub(r'[^0-9A-Za-z]+', '_', str(expected_name)).strip('_').lower()\n"
        "    for candidate_name, candidate_value in kwargs.items():\n"
        "        normalized_candidate = _normalize_tensor_name(str(candidate_name))\n"
        "        canonical_candidate = re.sub(r'[^0-9A-Za-z]+', '_', str(candidate_name)).strip('_').lower()\n"
        "        if (\n"
        "            normalized_candidate == normalized_expected_name\n"
        "            or canonical_candidate == canonical_expected_name\n"
        "            or normalized_candidate.endswith(normalized_expected_name)\n"
        "            or normalized_expected_name.endswith(normalized_candidate)\n"
        "            or canonical_candidate.endswith(canonical_expected_name)\n"
        "            or canonical_expected_name.endswith(canonical_candidate)\n"
        "        ):\n"
        "            return candidate_value\n"
        "    raise KeyError(str(expected_name))\n\n"
        "def _perm_cl_to_cf(rank: int) -> Optional[List[int]]:\n"
        "    if rank == 3:\n"
        "        return [0, 2, 1]\n"
        "    if rank == 4:\n"
        "        return [0, 3, 1, 2]\n"
        "    if rank == 5:\n"
        "        return [0, 4, 1, 2, 3]\n"
        "    return None\n\n"
        "def _perm_cf_to_cl(rank: int) -> Optional[List[int]]:\n"
        "    if rank == 3:\n"
        "        return [0, 2, 1]\n"
        "    if rank == 4:\n"
        "        return [0, 2, 3, 1]\n"
        "    if rank == 5:\n"
        "        return [0, 2, 3, 4, 1]\n"
        "    return None\n\n"
        "def _permute_shape(values: Sequence[int], perm: Sequence[int]) -> List[int]:\n"
        "    items = [int(v) for v in list(values)]\n"
        "    return [int(items[idx]) for idx in perm]\n\n"
        "def _align_tensor_to_target_shape(value: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    if target_shape is None:\n"
        "        return value\n"
        "    actual_shape = [int(v) for v in list(value.shape)]\n"
        "    target = [int(v) for v in list(target_shape)]\n"
        "    if actual_shape == target:\n"
        "        return value\n"
        "    perm = _perm_cl_to_cf(value.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == target:\n"
        "        return value.permute(*perm).contiguous()\n"
        "    perm_inv = _perm_cf_to_cl(value.ndim)\n"
        "    if perm_inv is not None and _permute_shape(actual_shape, perm_inv) == target:\n"
        "        return value.permute(*perm_inv).contiguous()\n"
        "    return value\n\n"
        "def _matches_target_except_axis(actual_shape: Sequence[int], target_shape: Sequence[int], axis: int) -> bool:\n"
        "    if len(list(actual_shape)) != len(list(target_shape)):\n"
        "        return False\n"
        "    for idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target_shape)):\n"
        "        if int(idx) == int(axis):\n"
        "            continue\n"
        "        if int(actual_dim) != int(target_dim):\n"
        "            return False\n"
        "    return True\n\n"
        "def _align_binary_inputs(x: torch.Tensor, y: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    if x.ndim != y.ndim:\n"
        "        return x, y\n"
        "    if [int(v) for v in list(x.shape)] == [int(v) for v in list(y.shape)]:\n"
        "        return x, y\n"
        "    try:\n"
        "        torch.broadcast_shapes(tuple(int(v) for v in x.shape), tuple(int(v) for v in y.shape))\n"
        "        return x, y\n"
        "    except Exception:\n"
        "        pass\n"
        "    perm = _perm_cl_to_cf(x.ndim)\n"
        "    if perm is None:\n"
        "        return x, y\n"
        "    x_shape = [int(v) for v in list(x.shape)]\n"
        "    y_shape = [int(v) for v in list(y.shape)]\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    if _permute_shape(y_shape, perm) == x_shape:\n"
        "        return x, y.permute(*perm).contiguous()\n"
        "    if _permute_shape(x_shape, perm) == y_shape:\n"
        "        return x.permute(*perm).contiguous(), y\n"
        "    if target is not None:\n"
        "        if _permute_shape(y_shape, perm) == target:\n"
        "            return x, y.permute(*perm).contiguous()\n"
        "        if _permute_shape(x_shape, perm) == target:\n"
        "            return x.permute(*perm).contiguous(), y\n"
        "    return x, y\n\n"
        "def _normalize_dim(dim: int, rank: int) -> int:\n"
        "    resolved = int(dim)\n"
        "    if resolved < 0:\n"
        "        resolved += int(rank)\n"
        "    return resolved\n\n"
        "def _coerce_scalar_axis(value: Any, *, device: torch.device) -> int:\n"
        "    if isinstance(value, torch.Tensor):\n"
        "        flat = value.to(dtype=torch.int64, device=device).reshape(-1)\n"
        "        if int(flat.numel()) == 0:\n"
        "            return 0\n"
        "        return int(flat[0].item())\n"
        "    return int(value)\n\n"
        "def _shape_list(value: Any) -> List[int]:\n"
        "    if isinstance(value, torch.Tensor):\n"
        "        return [int(v) for v in value.to(dtype=torch.int64).reshape(-1).tolist()]\n"
        "    if isinstance(value, np.ndarray):\n"
        "        return [int(v) for v in value.reshape(-1).tolist()]\n"
        "    return [int(v) for v in list(value)]\n\n"
        "def _to_torch_pad_arg(paddings: torch.Tensor) -> List[int]:\n"
        "    pads = paddings.to(dtype=torch.int64).reshape(-1, 2).tolist()\n"
        "    torch_pad: List[int] = []\n"
        "    for before, after in reversed(pads):\n"
        "        torch_pad.extend([int(before), int(after)])\n"
        "    return torch_pad\n\n"
        "def _infer_spatial_shape_for_transposed_conv2d(*, raw_output: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int]) -> Tuple[int, int]:\n"
        "    output_channels = int(raw_output.shape[1])\n"
        "    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]\n"
        "    if len(source) == 4:\n"
        "        if int(source[1]) == output_channels:\n"
        "            return int(source[2]), int(source[3])\n"
        "        if int(source[-1]) == output_channels:\n"
        "            return int(source[1]), int(source[2])\n"
        "    return int(source[-2]), int(source[-1])\n\n"
        "def _infer_spatial_shape_for_transposed_conv3d(*, raw_output: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int]) -> Tuple[int, int, int]:\n"
        "    output_channels = int(raw_output.shape[1])\n"
        "    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]\n"
        "    if len(source) == 5:\n"
        "        if int(source[1]) == output_channels:\n"
        "            return int(source[2]), int(source[3]), int(source[4])\n"
        "        if int(source[-1]) == output_channels:\n"
        "            return int(source[1]), int(source[2]), int(source[3])\n"
        "    return int(source[-3]), int(source[-2]), int(source[-1])\n\n"
        "def _apply_fused_activation(x: torch.Tensor, fused: str) -> torch.Tensor:\n"
        "    key = str(fused).upper()\n"
        "    if key in {'', 'NONE'}:\n"
        "        return x\n"
        "    if key == 'RELU':\n"
        "        return torch.relu(x)\n"
        "    if key == 'RELU6':\n"
        "        return torch.clamp(x, min=0.0, max=6.0)\n"
        "    if key == 'RELU_N1_TO_1':\n"
        "        return torch.clamp(x, min=-1.0, max=1.0)\n"
        "    if key == 'RELU_0_TO_1':\n"
        "        return torch.clamp(x, min=0.0, max=1.0)\n"
        "    if key == 'TANH':\n"
        "        return torch.tanh(x)\n"
        "    return x\n\n"
        "def _lookup_state_tensor(raw_state_dict: Dict[str, Any], tensor_name: str, storage_names: Dict[str, str]) -> torch.Tensor:\n"
        "    original_key = str(tensor_name)\n"
        "    storage_key = storage_names.get(original_key, _default_tensor_storage_name(original_key))\n"
        "    if original_key in raw_state_dict:\n"
        "        return torch.as_tensor(raw_state_dict[original_key])\n"
        "    if storage_key in raw_state_dict:\n"
        "        return torch.as_tensor(raw_state_dict[storage_key])\n"
        "    raise KeyError(original_key)\n\n"
        "def _copy_tensor_data(target: torch.Tensor, source: torch.Tensor) -> None:\n"
        "    target.data.copy_(source.to(device=target.device, dtype=target.dtype))\n\n"
        "def _validate_state_dict_keys(raw_state_dict: Dict[str, Any], storage_names: Dict[str, str], expected_tensor_names: Sequence[str]) -> None:\n"
        "    recognized_keys: Set[str] = set()\n"
        "    missing: List[str] = []\n"
        "    for tensor_name in expected_tensor_names:\n"
        "        storage_key = storage_names.get(str(tensor_name), _default_tensor_storage_name(str(tensor_name)))\n"
        "        if str(tensor_name) in raw_state_dict:\n"
        "            recognized_keys.add(str(tensor_name))\n"
        "            continue\n"
        "        if storage_key in raw_state_dict:\n"
        "            recognized_keys.add(storage_key)\n"
        "            continue\n"
        "        missing.append(str(tensor_name))\n"
        "    unexpected = sorted(str(key) for key in raw_state_dict.keys() if str(key) not in recognized_keys)\n"
        "    if len(missing) > 0 or len(unexpected) > 0:\n"
        "        raise RuntimeError(f'state_dict mismatch. missing={missing} unexpected={unexpected}')\n\n"
        "def _apply_binary(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, y: torch.Tensor, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    x, y = _align_binary_inputs(x, y, target_shape)\n"
        "    z = fn(x, y)\n"
        "    z = _align_tensor_to_target_shape(z, target_shape)\n"
        "    return _apply_fused_activation(z, fused)\n\n"
        "def _apply_concat(values: Sequence[torch.Tensor], axis: int, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    if any(int(value.ndim) == 0 for value in values):\n"
        "        values = [value.reshape(1) if int(value.ndim) == 0 else value for value in values]\n"
        "    rank = int(values[0].ndim)\n"
        "    resolved_axis = _normalize_dim(int(axis), rank)\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    if target is not None and len(target) == rank:\n"
        "        aligned_values: List[torch.Tensor] = []\n"
        "        for value in values:\n"
        "            actual = [int(v) for v in list(value.shape)]\n"
        "            chosen = value\n"
        "            if actual != target:\n"
        "                perm = _perm_cl_to_cf(value.ndim)\n"
        "                if perm is not None:\n"
        "                    permuted_shape = _permute_shape(actual, perm)\n"
        "                    if _matches_target_except_axis(permuted_shape, target, resolved_axis):\n"
        "                        chosen = value.permute(*perm).contiguous()\n"
        "            aligned_values.append(chosen)\n"
        "        values = aligned_values\n"
        "    y = torch.cat(list(values), dim=resolved_axis)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_conv2d(module: torch.nn.Module, x: torch.Tensor, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    expected_in_channels = int(module.in_channels)\n"
        "    if x.ndim == 4 and int(x.shape[1]) != expected_in_channels and int(x.shape[-1]) == expected_in_channels:\n"
        "        x = x.permute(0, 3, 1, 2).contiguous()\n"
        "    y = module(x)\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_transpose_conv2d(module: torch.nn.Module, x: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], fused: str) -> torch.Tensor:\n"
        "    weight = module.weight\n"
        "    if x.ndim == 4 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):\n"
        "        x = x.permute(0, 3, 1, 2).contiguous()\n"
        "    raw = module(x)\n"
        "    target_h, target_w = _infer_spatial_shape_for_transposed_conv2d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)\n"
        "    y = raw[..., :target_h, :target_w]\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_conv3d(module: torch.nn.Module, x: torch.Tensor, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    weight = module.weight\n"
        "    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[1]) and int(x.shape[-1]) == int(weight.shape[1]):\n"
        "        x = x.permute(0, 4, 1, 2, 3).contiguous()\n"
        "    y = module(x)\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_transpose_conv3d(module: torch.nn.Module, x: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], fused: str) -> torch.Tensor:\n"
        "    weight = module.weight\n"
        "    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):\n"
        "        x = x.permute(0, 4, 1, 2, 3).contiguous()\n"
        "    raw = module(x)\n"
        "    target_d, target_h, target_w = _infer_spatial_shape_for_transposed_conv3d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)\n"
        "    y = raw[..., :target_d, :target_h, :target_w]\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_softmax(x: torch.Tensor, axis: Optional[int], beta: float, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    resolved_axis = _normalize_dim(int(axis), x.ndim) if axis is not None else -1\n"
        "    if beta != 1.0:\n"
        "        x = x * beta\n"
        "    y = torch.softmax(x, dim=resolved_axis)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _normalize_axes(value: Any, rank: int) -> Optional[Tuple[int, ...]]:\n"
        "    if value is None:\n"
        "        return None\n"
        "    axes = _shape_list(value)\n"
        "    return tuple(sorted({_normalize_dim(int(v), rank) for v in axes}))\n\n"
        "def _reduce_sum(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.sum(x) if not keepdims else torch.sum(x).reshape([1] * x.ndim)\n"
        "    return torch.sum(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_mean(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.mean(x) if not keepdims else torch.mean(x).reshape([1] * x.ndim)\n"
        "    return torch.mean(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_max(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.amax(x, keepdim=keepdims)\n"
        "    return torch.amax(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_min(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.amin(x, keepdim=keepdims)\n"
        "    return torch.amin(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_prod(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        y = torch.prod(x)\n"
        "        return y if not keepdims else y.reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    for dim in sorted(axis, reverse=True):\n"
        "        result = torch.prod(result, dim=dim, keepdim=keepdims)\n"
        "    return result\n\n"
        "def _reduce_any(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        y = torch.any(x)\n"
        "        return y if not keepdims else y.reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    for dim in sorted(axis, reverse=True):\n"
        "        result = torch.any(result, dim=dim, keepdim=keepdims)\n"
        "    return result\n\n"
    )

    expected_tensor_names = [
        str(name) for name, tensor in model_ir.tensors.items() if isinstance(tensor.data, np.ndarray)
    ]
    tensor_storage_literal = repr({str(k): str(v) for k, v in tensor_storage_name_map.items()})
    module_init_block = "\n".join(f"        {line}" for line in module_init_lines)
    buffer_init_block = "\n".join(f"        {line}" for line in buffer_init_lines)
    state_load_block = "\n".join(f"            {line}" for line in weight_load_lines + buffer_load_lines)
    forward_kwargs_lines: List[str] = []
    forward_args_lines: List[str] = []
    for input_index, input_name in enumerate(model_ir.inputs):
        input_var = tensor_var_names[str(input_name)]
        forward_kwargs_lines.append(
            f"            {input_var} = _resolve_named_input_value(kwargs, {str(input_name)!r})"
        )
        forward_args_lines.append(
            f"            {input_var} = args[{input_index}]"
        )
    forward_block = "\n".join(f"        {line}" for line in forward_lines)
    forward_kwargs_block = "\n".join(forward_kwargs_lines) if len(forward_kwargs_lines) > 0 else "            pass"
    forward_args_block = "\n".join(forward_args_lines) if len(forward_args_lines) > 0 else "            pass"
    outputs_expr = ", ".join(_tensor_expr(str(name)) for name in model_ir.outputs)

    model_source = (
        "from __future__ import annotations\n\n"
        "import copy\n"
        "import json\n"
        "from pathlib import Path\n"
        "import re\n"
        "from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple\n\n"
        "import numpy as np\n"
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        f"{helper_source}"
        "PACKAGE_DIR = Path(__file__).resolve().parent\n"
        f"MODEL_METADATA = json.loads({metadata_literal})\n\n"
        "class Model(torch.nn.Module):\n"
        "    def __init__(self, *, device: str | None = None, eval_mode: bool = True, load_weights: bool = True, metadata: dict | None = None):\n"
        "        super().__init__()\n"
        "        self._metadata = copy.deepcopy(MODEL_METADATA if metadata is None else metadata)\n"
        "        self.input_names = list(self._metadata.get('inputs', []))\n"
        "        self.output_names = list(self._metadata.get('outputs', []))\n"
        f"        self.tensor_storage_names = dict({tensor_storage_literal})\n"
        f"{module_init_block}\n"
        f"{buffer_init_block}\n"
        "        if load_weights:\n"
        "            raw_state_dict = torch.load(PACKAGE_DIR / 'state_dict.pth', map_location=device or 'cpu')\n"
        f"            _validate_state_dict_keys(raw_state_dict, self.tensor_storage_names, {repr(expected_tensor_names)})\n"
        f"{state_load_block}\n"
        "        elif device is not None:\n"
        "            self.to(device)\n"
        "        if device is not None and load_weights:\n"
        "            self.to(device)\n"
        "        if eval_mode:\n"
        "            self.eval()\n\n"
        "    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Any:\n"
        "        if len(args) > 0 and len(kwargs) > 0:\n"
        "            raise RuntimeError('Use either positional inputs or keyword inputs, not both.')\n"
        "        if len(kwargs) > 0:\n"
        f"{forward_kwargs_block}\n"
        "        else:\n"
        f"            if len(args) != {len(model_ir.inputs)}:\n"
        "                raise RuntimeError(f'Input arity mismatch. expected={len(self.input_names)} actual={len(args)}')\n"
        f"{forward_args_block}\n"
        f"{forward_block}\n"
    )
    if len(model_ir.outputs) == 1:
        model_source += (
            f"        return {outputs_expr}\n\n"
            "    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:\n"
            f"        return {{{str(model_ir.outputs[0])!r}: self.forward(*args, **kwargs)}}\n\n"
        )
    else:
        model_source += (
            f"        return ({outputs_expr})\n\n"
            "    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:\n"
            "        result = self.forward(*args, **kwargs)\n"
            f"        return {{name: value for name, value in zip({repr([str(v) for v in model_ir.outputs])}, result)}}\n\n"
        )
    model_source += (
        "def load_model(device: str | None = None, eval_mode: bool = True) -> Model:\n"
        "    return Model(device=device, eval_mode=eval_mode)\n"
    )
    (package_dir / "model.py").write_text(model_source, encoding="utf-8")


def export_pytorch_package_from_tflite_artifact(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    tflite_file_path: str,
) -> str:
    if not os.path.exists(tflite_file_path):
        raise ModelIRPyTorchExportError(
            f"TFLite-backed PyTorch package export requires an existing float32 TFLite file. path={tflite_file_path}"
        )

    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)

    package_tflite_name = "model_float32.tflite"
    shutil.copyfile(
        str(tflite_file_path),
        os.path.join(output_folder_path, package_tflite_name),
    )
    metadata = _build_tflite_backed_metadata_payload(
        model_ir=model_ir,
        tflite_file_name=package_tflite_name,
    )
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)


def export_pytorch_package_from_saved_model_artifact(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    saved_model_path: str,
) -> str:
    if not os.path.exists(saved_model_path):
        raise ModelIRPyTorchExportError(
            f"SavedModel-backed PyTorch package export requires an existing SavedModel directory. path={saved_model_path}"
        )

    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)

    package_saved_model_dir = os.path.join(output_folder_path, "saved_model")
    if os.path.exists(package_saved_model_dir):
        shutil.rmtree(package_saved_model_dir)
    shutil.copytree(str(saved_model_path), package_saved_model_dir)
    metadata = _build_saved_model_backed_metadata_payload(
        model_ir=model_ir,
        saved_model_dir_name="saved_model",
    )
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)


def _should_prefer_tflite_backed_package(model_ir: ModelIR) -> bool:
    op_types = [str(op.op_type) for op in model_ir.operators]
    recurrent_or_control_ops = {
        "WHILE",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if any(op_type in recurrent_or_control_ops for op_type in op_types):
        return False
    has_length_like_input = False
    for input_name in model_ir.inputs:
        canonical = re.sub(r"[^0-9a-z]+", "_", str(input_name).lower()).strip("_")
        if canonical.endswith(("length", "lengths", "len", "lens", "seq_len", "seq_lens")):
            has_length_like_input = True
            break
    if has_length_like_input:
        return False
    if any(op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} for op_type in op_types):
        return True
    for op in model_ir.operators:
        if str(op.op_type) != "SOFTMAX" or len(op.inputs) == 0:
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        if input_tensor is None:
            continue
        if is_channel_first_logical_layout(normalize_logical_layout(input_tensor.logical_layout)):
            return True
    conv_like_count = sum(
        1
        for op_type in op_types
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
    )
    strided_slice_count = sum(1 for op_type in op_types if op_type == "STRIDED_SLICE")
    concat_count = sum(1 for op_type in op_types if op_type == "CONCATENATION")
    resize_count = sum(
        1
        for op_type in op_types
        if op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}
    )
    split_count = sum(1 for op_type in op_types if op_type == "SPLIT")
    softmax_count = sum(1 for op_type in op_types if op_type == "SOFTMAX")
    nhwc_named_tensor_count = sum(
        1
        for tensor_name in model_ir.tensors.keys()
        if str(tensor_name).lower().endswith(("_nhwc", "_nwc", "_ndhwc"))
    )
    has_rank3_channel_first_output = any(
        len(list(model_ir.tensors[str(output_name)].shape)) == 3
        and normalize_logical_layout(model_ir.tensors[str(output_name)].logical_layout) == "NCW"
        for output_name in model_ir.outputs
        if str(output_name) in model_ir.tensors
    )
    if (
        has_rank3_channel_first_output
        and conv_like_count >= 20
        and strided_slice_count >= 4
        and concat_count >= 4
    ):
        return True
    if (
        conv_like_count >= 40
        and nhwc_named_tensor_count >= 40
        and (resize_count >= 4 or softmax_count >= 1 or split_count >= 1)
    ):
        return True
    if (
        conv_like_count >= 60
        and nhwc_named_tensor_count >= 80
        and resize_count >= 2
    ):
        return True
    if (
        conv_like_count >= 15
        and nhwc_named_tensor_count >= 30
        and resize_count >= 3
    ):
        return True
    return False


def _should_prefer_saved_model_backed_package(model_ir: ModelIR) -> bool:
    return _should_prefer_tflite_backed_package(model_ir)


def _merge_reference_public_boundary_metadata(
    *,
    imported_model_ir: ModelIR,
    reference_model_ir: Optional[ModelIR],
    reference_onnx_graph: Optional[Any] = None,
) -> None:
    if reference_model_ir is None:
        return
    imported_model_ir.inputs = [str(v) for v in list(reference_model_ir.inputs)]
    imported_model_ir.outputs = [str(v) for v in list(reference_model_ir.outputs)]
    boundary_shape_map = reference_model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
    public_layout_map = reference_model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    recurrent_public_boundary_context = any(
        token in str(op.op_type)
        for op in reference_model_ir.operators
        for token in ("GRU", "LSTM", "RNN")
    )
    if not recurrent_public_boundary_context and reference_onnx_graph is not None:
        graph = getattr(reference_onnx_graph, "graph", None)
        if graph is not None:
            recurrent_public_boundary_context = any(
                str(node.op_type) in {"GRU", "LSTM", "RNN"}
                for node in list(graph.node)
            )
    for tensor_name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs):
        ref_tensor = reference_model_ir.tensors.get(str(tensor_name), None)
        imported_tensor = imported_model_ir.tensors.get(str(tensor_name), None)
        if ref_tensor is None or imported_tensor is None:
            continue
        imported_tensor.shape_signature = [int(v) for v in list(ref_tensor.shape_signature or ref_tensor.shape)]
        inferred_public_layout = normalize_logical_layout(ref_tensor.logical_layout)
        if recurrent_public_boundary_context and len(list(ref_tensor.shape)) == 3:
            inferred_public_layout = "NWC"
        imported_tensor.logical_layout = inferred_public_layout
    imported_model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        str(name): [int(v) for v in list(boundary_shape_map.get(str(name), reference_model_ir.tensors[str(name)].shape))]
        for name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
        if str(name) in reference_model_ir.tensors
    }
    imported_model_ir.metadata["onnx_public_layout_map"] = {
        str(name): (
            "NWC"
            if recurrent_public_boundary_context and len(list(reference_model_ir.tensors[str(name)].shape)) == 3
            else normalize_logical_layout(
                public_layout_map.get(
                    str(name),
                    reference_model_ir.tensors[str(name)].logical_layout,
                )
            )
        )
        for name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
        if str(name) in reference_model_ir.tensors
    }


def _try_export_native_package_from_tflite_import(
    *,
    output_folder_path: str,
    fallback_tflite_path: str,
    reference_model_ir: Optional[ModelIR] = None,
    reference_onnx_graph: Optional[Any] = None,
) -> Optional[str]:
    if not os.path.exists(str(fallback_tflite_path)):
        return None
    imported_model_ir = import_model_ir_from_tflite(
        tflite_file_path=str(fallback_tflite_path),
    )
    _merge_reference_public_boundary_metadata(
        imported_model_ir=imported_model_ir,
        reference_model_ir=reference_model_ir,
        reference_onnx_graph=reference_onnx_graph,
    )
    return export_pytorch_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=output_folder_path,
        fallback_tflite_path=None,
        fallback_onnx_graph=None,
        fallback_saved_model_path=None,
        fallback_tflite_has_custom_ops=False,
    )


def _try_export_runtime_wrapper_package_from_tflite_import(
    *,
    output_folder_path: str,
    fallback_tflite_path: str,
) -> Optional[str]:
    if not os.path.exists(str(fallback_tflite_path)):
        return None
    imported_model_ir = import_model_ir_from_tflite(
        tflite_file_path=str(fallback_tflite_path),
    )
    if not _supports_runtime_wrapper_model_ir(imported_model_ir):
        return None
    return _export_runtime_wrapper_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=output_folder_path,
    )


def export_pytorch_package_from_model_ir(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    fallback_tflite_path: Optional[str] = None,
    fallback_onnx_graph: Optional[Any] = None,
    fallback_saved_model_path: Optional[str] = None,
    fallback_tflite_has_custom_ops: bool = False,
) -> str:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "PyTorch export requires `torch` to be installed."
        ) from ex

    try:
        if (
            fallback_saved_model_path is not None
            and str(fallback_saved_model_path).strip() != ""
            and _should_prefer_saved_model_backed_package(model_ir)
        ):
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path),
            )
        if (
            fallback_saved_model_path is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and not bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            return export_pytorch_package_from_tflite_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                tflite_file_path=str(fallback_tflite_path),
            )
        if (
            fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            runtime_wrapper_path = _try_export_runtime_wrapper_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
            )
            if runtime_wrapper_path is not None:
                return runtime_wrapper_path

        normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)
        _ensure_no_custom_ops(normalized)
        _ensure_supported_ops(normalized)
        tensor_storage_name_map = _make_tensor_storage_name_map(normalized)

        os.makedirs(output_folder_path, exist_ok=True)
        metadata = _build_metadata_payload(normalized)
        metadata["tensor_storage_names"] = dict(tensor_storage_name_map)
        _write_generated_package_common_files(output_folder_path)
        try:
            _write_native_model_file(
                output_folder_path,
                model_ir=normalized,
                metadata=metadata,
                tensor_storage_name_map=tensor_storage_name_map,
            )
        except ModelIRPyTorchExportError as ex:
            if not _is_direct_codegen_unsupported_error(ex):
                raise
            # Keep torch-kernel-backed packages native when runtime kernels
            # support the graph, even if direct Python codegen does not yet.
            _write_wrapper_model_file(output_folder_path)
        metadata_path = os.path.join(output_folder_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        state_dict: Dict[str, Any] = {}
        for tensor_name, tensor in normalized.tensors.items():
            if not isinstance(tensor.data, np.ndarray):
                continue
            dtype_name = str(tensor.dtype).upper()
            if dtype_name not in {"BOOL", "INT8", "INT16", "INT32", "INT64", "UINT8", "FLOAT16", "FLOAT32", "FLOAT64"}:
                raise ModelIRPyTorchExportError(
                    f"Unsupported tensor dtype for PyTorch export: tensor={tensor_name} dtype={tensor.dtype}"
                )
            storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
            state_dict[storage_name] = torch.as_tensor(np.asarray(tensor.data))
        torch.save(state_dict, os.path.join(output_folder_path, "state_dict.pth"))
        return str(output_folder_path)
    except Exception:
        string_config = _extract_string_normalizer_config_from_onnx_graph(
            fallback_onnx_graph,
        )
        if string_config is not None:
            return export_pytorch_package_from_string_normalizer_onnx(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                onnx_graph=fallback_onnx_graph,
            )
        if fallback_saved_model_path is not None and str(fallback_saved_model_path).strip() != "":
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path),
            )
        if fallback_tflite_path is None or str(fallback_tflite_path).strip() == "":
            raise
        try:
            runtime_wrapper_path = _try_export_runtime_wrapper_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
            )
            if runtime_wrapper_path is not None:
                return runtime_wrapper_path
        except Exception:
            pass
        if not bool(fallback_tflite_has_custom_ops):
            try:
                return _try_export_native_package_from_tflite_import(
                    output_folder_path=output_folder_path,
                    fallback_tflite_path=str(fallback_tflite_path),
                    reference_model_ir=model_ir,
                    reference_onnx_graph=fallback_onnx_graph,
                )
            except Exception:
                pass
        if (
            not bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            return export_pytorch_package_from_tflite_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                tflite_file_path=str(fallback_tflite_path),
            )
        return export_pytorch_package_from_tflite_artifact(
            model_ir=model_ir,
            output_folder_path=output_folder_path,
            tflite_file_path=str(fallback_tflite_path),
        )
