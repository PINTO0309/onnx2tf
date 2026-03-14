from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import keyword
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple, cast

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
    logical_layout_rank,
    logical_layout_permutation,
    normalize_logical_layout,
    rewrite_axis_for_layout,
    validate_model_ir_layout_annotations,
)
from onnx2tf.tflite_builder.pytorch_package_runtime import (
    SUPPORTED_TORCH_KERNEL_OP_TYPES,
)
from onnx2tf.tflite_builder.split_planner import (
    rewrite_model_ir_unroll_recurrent_ops,
)
from onnx2tf.tflite_builder.tflite_importer import (
    import_model_ir_from_tflite,
)


from .common import ModelIRPyTorchExportError, _NativeCodegenBindings, _NativeCodegenState, _NativeModelFileWriterContext, _broadcast_shapes_relaxed, _compose_axis_permutations, _is_all_ones_shape, _make_unique_identifier, _pad_output_matches_pre_permuted_input, _perm_cf_to_cl, _perm_cl_to_cf, _permute_shape, _preferred_reshape_target_values, _product_expr, _read_transpose_perm, _shape_can_broadcast_to_target_relaxed, _shape_lists_equal, _shape_lists_equal_relaxed
from .layout_normalization import _can_emit_direct_torch_reshape_shape, _collect_feature_last_sequence_tensor_names, _is_inconsistent_same_layout_transpose, _is_inconsistent_standard_layout_transpose, _is_reshape_only_residual_layout_bridge_transpose, _is_standard_channel_layout_permutation, _should_emit_channel_last_depth_to_space, _should_emit_channel_last_space_to_depth

def _prepare_native_codegen_state(
    context: _NativeModelFileWriterContext,
) -> _NativeCodegenState:
    return _NativeCodegenState(context=context)

def _build_native_codegen_bindings(
    state: _NativeCodegenState,
) -> _NativeCodegenBindings:
    return _NativeCodegenBindings()

def _build_native_constant_aliases(
    state: _NativeCodegenState,
    bindings: _NativeCodegenBindings,
) -> None:
    _ = state
    _ = bindings

def _emit_native_forward_lines(
    state: _NativeCodegenState,
    bindings: _NativeCodegenBindings,
) -> None:
    _ = bindings
    state.load_specs_result = _write_native_model_file_codegen_core_body_main_inner_legacy_impl(
        state.context
    )

def _finalize_native_codegen(
    state: _NativeCodegenState,
    bindings: _NativeCodegenBindings,
) -> List[Tuple[str, str]]:
    _ = bindings
    return [] if state.load_specs_result is None else list(state.load_specs_result)

def _assemble_native_model_source(
    *,
    model_ir: ModelIR,
    runtime_import_block: str,
    sequence_rnn_helper_source: str,
    sequence_lstm_helper_source: str,
    affine_layer_norm_source: str,
    named_encoder_class_source: str,
    buffer_annotation_block: str,
    module_init_block: str,
    init_constants_call: str,
    constant_buffer_alias_init_call: str,
    init_constants_method: str,
    constant_buffer_alias_method_source: str,
    constant_buffer_alias_load_state_dict_source: str,
    nms_method_source: str,
    stage_methods_source: str,
    forward_signature: str,
    forward_block: str,
    outputs_expr: str,
    forward_kwargs_block: str,
    forward_args_block: str,
    forward_named_call_args: str,
) -> str:
    model_source = _assemble_native_model_source(
        model_ir=model_ir,
        runtime_import_block=runtime_import_block,
        sequence_rnn_helper_source=sequence_rnn_helper_source,
        sequence_lstm_helper_source=sequence_lstm_helper_source,
        affine_layer_norm_source=affine_layer_norm_source,
        named_encoder_class_source=named_encoder_class_source,
        buffer_annotation_block=buffer_annotation_block,
        module_init_block=module_init_block,
        init_constants_call=init_constants_call,
        constant_buffer_alias_init_call=constant_buffer_alias_init_call,
        init_constants_method=init_constants_method,
        constant_buffer_alias_method_source=constant_buffer_alias_method_source,
        constant_buffer_alias_load_state_dict_source=constant_buffer_alias_load_state_dict_source,
        nms_method_source=nms_method_source,
        stage_methods_source=stage_methods_source,
        forward_signature=forward_signature,
        forward_block=forward_block,
        outputs_expr=outputs_expr,
        forward_kwargs_block=forward_kwargs_block,
        forward_args_block=forward_args_block,
        forward_named_call_args=forward_named_call_args,
    )
    return model_source

def _extract_statement_assignments(statement: ast.stmt) -> List[str]:
    names: List[str] = []

    def _walk_target(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.append(str(target.id))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                _walk_target(item)

    if isinstance(statement, ast.Assign):
        for target in statement.targets:
            _walk_target(target)
    elif isinstance(statement, ast.AnnAssign):
        _walk_target(statement.target)
    return names

def _extract_statement_loads(statement: ast.stmt) -> List[str]:
    names: List[str] = []
    seen: Set[str] = set()
    for node in ast.walk(statement):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and str(node.id) not in seen:
            seen.add(str(node.id))
            names.append(str(node.id))
    return names

def _shape_literal(values: Sequence[int]) -> str:
    return repr(tuple(int(v) for v in list(values)))

def _remap_axis_values_through_permutation(
    values: Sequence[int],
    perm: Sequence[int],
) -> List[int]:
    remapped = [0] * len(list(perm))
    for output_axis, input_axis in enumerate(list(perm)):
        remapped[int(input_axis)] = int(values[output_axis])
    return [int(v) for v in list(remapped)]

def _remap_mask_bits_through_permutation(
    mask: int,
    perm: Sequence[int],
) -> int:
    remapped_mask = 0
    for output_axis, input_axis in enumerate(list(perm)):
        if int(mask) & (1 << int(output_axis)):
            remapped_mask |= 1 << int(input_axis)
    return int(remapped_mask)

def _add_synthetic_tensor_to_model_ir(
    *,
    model_ir: ModelIR,
    base_name: str,
    data: np.ndarray,
    dtype: str,
    synthetic_tensor_serial_ref: List[int],
) -> str:
    candidate = str(base_name)
    while candidate in model_ir.tensors:
        synthetic_tensor_serial_ref[0] += 1
        candidate = f"{base_name}_{synthetic_tensor_serial_ref[0]}"
    array = np.asarray(data)
    model_ir.tensors[candidate] = TensorIR(
        name=candidate,
        dtype=str(dtype),
        shape=[int(v) for v in list(array.shape)],
        shape_signature=[int(v) for v in list(array.shape)],
        data=array,
    )
    return candidate

def _require_constant_array_from_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    context: str,
) -> np.ndarray:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        raise ModelIRPyTorchExportError(
            f"Native PyTorch-like model.py codegen requires constant tensor data for {context}. "
            f"tensor={tensor_name}"
        )
    return np.asarray(tensor.data)

def _sequence_lstm_bias_array_for_model_ir(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    indices: Sequence[int],
    hidden_size: int,
    dtype: str,
    base_name: str,
    synthetic_tensor_serial_ref: List[int],
) -> str:
    bias_names = [_sequence_lstm_input_name(op, int(index)) for index in list(indices)]
    if all(name == "" for name in bias_names):
        return _add_synthetic_tensor_to_model_ir(
            model_ir=model_ir,
            base_name=base_name,
            data=np.zeros((4 * int(hidden_size),), dtype=np.float32),
            dtype=str(dtype),
            synthetic_tensor_serial_ref=synthetic_tensor_serial_ref,
        )
    if any(name == "" for name in bias_names):
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen requires LSTM gate biases to be either all present or all omitted."
        )
    concatenated = np.concatenate(
        [
            _require_constant_array_from_model_ir(
                model_ir=model_ir,
                tensor_name=name,
                context=f"LSTM bias gate {index}",
            ).reshape(-1)
            for index, name in enumerate(bias_names)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return _add_synthetic_tensor_to_model_ir(
        model_ir=model_ir,
        base_name=base_name,
        data=concatenated,
        dtype=str(dtype),
        synthetic_tensor_serial_ref=synthetic_tensor_serial_ref,
    )

def _fold_single_consumer_public_input_bridge_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    consumer_index: Dict[str, List[int]],
    public_layout_bridge_tensor_names: Set[str],
    public_input_names: Set[str],
    tensor_name: str,
    downstream_permute: Optional[Sequence[int]],
) -> Tuple[str, Optional[List[int]], Optional[int]]:
    producer_idx = producer_index.get(str(tensor_name), None)
    resolved_downstream_permute = (
        [int(v) for v in list(downstream_permute)]
        if downstream_permute is not None
        else None
    )
    if producer_idx is None:
        return str(tensor_name), resolved_downstream_permute, None
    producer_op = model_ir.operators[int(producer_idx)]
    if str(producer_op.op_type) != "TRANSPOSE" or len(producer_op.outputs) != 1 or len(producer_op.inputs) < 1:
        return str(tensor_name), resolved_downstream_permute, None
    bridge_output_name = str(producer_op.outputs[0])
    bridge_input_name = str(producer_op.inputs[0])
    if (
        bridge_output_name not in public_layout_bridge_tensor_names
        and not bridge_output_name.endswith("_public_layout_bridge")
    ):
        return str(tensor_name), resolved_downstream_permute, None
    if bridge_input_name not in public_input_names:
        return str(tensor_name), resolved_downstream_permute, None
    if len(consumer_index.get(bridge_output_name, [])) != 1:
        return str(tensor_name), resolved_downstream_permute, None
    bridge_perm = _read_transpose_perm(model_ir, producer_op)
    if bridge_perm is None:
        return str(tensor_name), resolved_downstream_permute, None
    composed_perm = _compose_axis_permutations(
        bridge_perm,
        downstream_permute,
    )
    return bridge_input_name, composed_perm, int(producer_idx)

def _match_single_consumer_layout_bridge_transpose_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    tensor_name: str,
    required_output_layout: Optional[str] = None,
) -> Optional[Tuple[str, int]]:
    consumer_indices = consumer_index.get(str(tensor_name), [])
    if len(consumer_indices) != 1:
        return None
    bridge_op_idx = int(consumer_indices[0])
    bridge_op = model_ir.operators[bridge_op_idx]
    if str(bridge_op.op_type) != "TRANSPOSE" or len(bridge_op.outputs) != 1:
        return None
    input_tensor = model_ir.tensors.get(str(tensor_name), None)
    output_name = str(bridge_op.outputs[0])
    output_tensor = model_ir.tensors.get(output_name, None)
    if input_tensor is None or output_tensor is None:
        return None
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if (
        input_layout == LOGICAL_LAYOUT_UNKNOWN
        or output_layout == LOGICAL_LAYOUT_UNKNOWN
        or input_layout == output_layout
    ):
        return None
    if required_output_layout is not None and output_layout != normalize_logical_layout(required_output_layout):
        return None
    expected_perm = logical_layout_permutation(
        source_layout=input_layout,
        target_layout=output_layout,
    )
    actual_perm = _read_transpose_perm(model_ir, bridge_op)
    if expected_perm is None or actual_perm is None:
        return None
    if [int(v) for v in list(expected_perm)] != [int(v) for v in list(actual_perm)]:
        return None
    return output_name, bridge_op_idx

def _target_shape_values_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    if tensor.shape_signature is not None:
        signature = [int(v) for v in list(tensor.shape_signature)]
        if len(signature) == len(list(tensor.shape)):
            return signature
    return [int(v) for v in list(tensor.shape)]

def _target_shape_literal_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> str:
    target_shape = _target_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    if target_shape is None:
        return "None"
    return repr([int(v) for v in list(target_shape)])

def _resize_target_shape_literal_for_model_ir(
    *,
    model_ir: ModelIR,
    output_name: str,
    input_name: str,
) -> str:
    output_tensor = model_ir.tensors.get(str(output_name), None)
    input_tensor = model_ir.tensors.get(str(input_name), None)
    if output_tensor is None:
        return "None"
    target_shape = [int(v) for v in list(output_tensor.shape)]
    if input_tensor is None:
        return repr(target_shape)
    input_shape = [int(v) for v in list(input_tensor.shape)]
    if target_shape == input_shape:
        return repr(target_shape)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    if (
        len(input_shape) == 4
        and len(target_shape) == 4
        and is_channel_first_logical_layout(input_layout)
        and is_channel_first_logical_layout(output_layout)
    ):
        if int(target_shape[1]) != int(input_shape[1]) and int(target_shape[-1]) == int(input_shape[1]):
            return repr([int(input_shape[0]), int(input_shape[1]), int(target_shape[1]), int(target_shape[2])])
        if int(target_shape[1]) != int(input_shape[-1]) and int(target_shape[-1]) == int(input_shape[-1]):
            return repr([int(input_shape[0]), int(input_shape[-1]), int(target_shape[1]), int(target_shape[2])])
    return repr(target_shape)

def _tensor_shape_list_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    return [int(v) for v in list(tensor.shape)]

def _rank4_channel_first_shape_for_tensor_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
) -> Optional[List[int]]:
    tensor_shape = _tensor_shape_list_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    if tensor_shape is None or len(tensor_shape) != 4:
        return None
    tensor = model_ir.tensors.get(str(tensor_name), None)
    is_nhwc_named = "_nhwc" in str(tensor_name)
    if (
        str(tensor_name) in channel_first_tensor_expr_aliases
        and tensor is not None
        and (
            is_nhwc_named
            or normalize_logical_layout(tensor.logical_layout) in {
                LOGICAL_LAYOUT_UNKNOWN,
                "NHWC",
            }
        )
    ):
        return [int(tensor_shape[0]), int(tensor_shape[3]), int(tensor_shape[1]), int(tensor_shape[2])]
    if tensor is not None and is_nhwc_named:
        return [int(tensor_shape[0]), int(tensor_shape[3]), int(tensor_shape[1]), int(tensor_shape[2])]
    if tensor is not None and is_channel_last_logical_layout(
        normalize_logical_layout(tensor.logical_layout)
    ):
        return [int(tensor_shape[0]), int(tensor_shape[3]), int(tensor_shape[1]), int(tensor_shape[2])]
    return [int(v) for v in list(tensor_shape)]

def _channel_first_shape_for_tensor_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
) -> Optional[List[int]]:
    tensor_shape = _tensor_shape_list_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    if tensor_shape is None:
        return None
    rank = len(list(tensor_shape))
    if rank not in {3, 4, 5}:
        return [int(v) for v in list(tensor_shape)]
    tensor = model_ir.tensors.get(str(tensor_name), None)
    perm_to_cf = _perm_cl_to_cf(rank)
    is_nhwc_named = rank == 4 and "_nhwc" in str(tensor_name)
    if (
        str(tensor_name) in channel_first_tensor_expr_aliases
        and tensor is not None
        and (
            is_nhwc_named
            or
            is_channel_last_logical_layout(
                normalize_logical_layout(tensor.logical_layout)
            )
            or normalize_logical_layout(tensor.logical_layout) == LOGICAL_LAYOUT_UNKNOWN
        )
        and perm_to_cf is not None
    ):
        permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
        if permuted_shape is not None:
            return [int(v) for v in list(permuted_shape)]
    if (
        tensor is not None
        and is_channel_last_logical_layout(
            normalize_logical_layout(tensor.logical_layout)
        )
        and perm_to_cf is not None
    ):
        permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
        if permuted_shape is not None:
            return [int(v) for v in list(permuted_shape)]
    if is_nhwc_named and perm_to_cf is not None:
        permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
        if permuted_shape is not None:
            return [int(v) for v in list(permuted_shape)]
    return [int(v) for v in list(tensor_shape)]

def _channel_first_concat_input_expr_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
    tensor_expr_fn: Callable[[str], str],
) -> Optional[str]:
    alias_expr = channel_first_tensor_expr_aliases.get(str(tensor_name), None)
    if alias_expr is not None:
        return str(alias_expr)
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    if is_channel_first_logical_layout(tensor_layout):
        return tensor_expr_fn(str(tensor_name))
    return None

def _can_fold_channel_last_alias_slice_consumer_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    expected_input_name: str,
) -> bool:
    op_type = str(op.op_type)
    if op_type == "SLICE":
        return (
            len(op.inputs) >= 3
            and str(op.inputs[0]) == str(expected_input_name)
            and _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None)) is not None
            and _constant_int_list(model_ir.tensors.get(str(op.inputs[2]), None)) is not None
        )
    if op_type == "STRIDED_SLICE":
        options = dict(op.options)
        return (
            len(op.inputs) >= 4
            and str(op.inputs[0]) == str(expected_input_name)
            and _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None)) is not None
            and _constant_int_list(model_ir.tensors.get(str(op.inputs[2]), None)) is not None
            and _constant_int_list(model_ir.tensors.get(str(op.inputs[3]), None)) is not None
            and int(options.get("ellipsisMask", 0)) == 0
            and int(options.get("newAxisMask", 0)) == 0
            and int(options.get("shrinkAxisMask", 0)) == 0
        )
    return False

def _is_valid_concat_axis_for_channel_first_shapes_for_codegen(
    *,
    input_shapes: Sequence[Sequence[int]],
    output_shape: Sequence[int],
    axis: int,
) -> bool:
    output_items = [int(v) for v in list(output_shape)]
    rank = len(output_items)
    if axis < 0 or axis >= rank:
        return False
    expected_axis_extent = 0
    axis_extent_static = int(output_items[axis]) > 0
    for input_shape in input_shapes:
        input_items = [int(v) for v in list(input_shape)]
        if len(input_items) != rank:
            return False
        for dim_index, (input_dim, output_dim) in enumerate(zip(input_items, output_items)):
            if dim_index == axis:
                continue
            if int(input_dim) > 0 and int(output_dim) > 0 and int(input_dim) != int(output_dim):
                return False
        if int(input_items[axis]) <= 0:
            axis_extent_static = False
        else:
            expected_axis_extent += int(input_items[axis])
    if axis_extent_static:
        return expected_axis_extent == int(output_items[axis])
    return True

def _resolve_concat_axis_for_channel_first_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    channel_first_shape_for_tensor_fn: Callable[[str], Optional[List[int]]],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
) -> Optional[Tuple[int, List[int], List[int]]]:
    if len(op.outputs) != 1:
        return None
    input_shapes_cf: List[List[int]] = []
    for input_name in op.inputs:
        input_shape_cf = channel_first_shape_for_tensor_fn(str(input_name))
        if input_shape_cf is None:
            return None
        input_shapes_cf.append([int(v) for v in list(input_shape_cf)])
    stored_output_shape = tensor_shape_list_fn(str(op.outputs[0]))
    if stored_output_shape is None:
        return None
    rank = len(list(stored_output_shape))
    axis = int(op.options.get("axis", 0))
    if axis < 0:
        axis += int(rank)
    candidate_output_specs: List[Tuple[List[int], List[int]]] = [
        (
            [int(v) for v in list(stored_output_shape)],
            [int(v) for v in list(range(rank))],
        ),
    ]
    perm_to_cf = _perm_cl_to_cf(rank)
    perm_from_cf = _perm_cf_to_cl(rank)
    if perm_to_cf is not None and perm_from_cf is not None:
        permuted_output_shape = _permute_shape(stored_output_shape, perm_to_cf)
        if permuted_output_shape is not None:
            candidate_spec = (
                [int(v) for v in list(permuted_output_shape)],
                [int(v) for v in list(perm_from_cf)],
            )
            if candidate_spec not in candidate_output_specs:
                candidate_output_specs.append(candidate_spec)
    if perm_from_cf is not None and perm_to_cf is not None:
        permuted_output_shape = _permute_shape(stored_output_shape, perm_from_cf)
        if permuted_output_shape is not None:
            candidate_spec = (
                [int(v) for v in list(permuted_output_shape)],
                [int(v) for v in list(perm_to_cf)],
            )
            if candidate_spec not in candidate_output_specs:
                candidate_output_specs.append(candidate_spec)
    candidate_axes: List[int] = []
    if 0 <= int(axis) < int(rank):
        candidate_axes.append(int(axis))
    if perm_to_cf is not None:
        mapped_axis = next(
            (int(index) for index, source_axis in enumerate(perm_to_cf) if int(source_axis) == int(axis)),
            None,
        )
        if mapped_axis is not None and int(mapped_axis) not in candidate_axes:
            candidate_axes.append(int(mapped_axis))
    for output_shape_cf, perm_from_candidate in candidate_output_specs:
        for candidate_axis in candidate_axes:
            if _is_valid_concat_axis_for_channel_first_shapes_for_codegen(
                input_shapes=input_shapes_cf,
                output_shape=output_shape_cf,
                axis=int(candidate_axis),
            ):
                return (
                    int(candidate_axis),
                    [int(v) for v in list(output_shape_cf)],
                    [int(v) for v in list(perm_from_candidate)],
                )
    return None

def _can_keep_channel_first_slice_output_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    output_name: str,
    resolve_concat_axis_for_channel_first_fn: Callable[[OperatorIR], Optional[Tuple[int, List[int], List[int]]]],
) -> bool:
    consumer_indices = consumer_index.get(str(output_name), [])
    if len(consumer_indices) == 0:
        return False
    for consumer_idx in consumer_indices:
        consumer_op = model_ir.operators[int(consumer_idx)]
        if str(consumer_op.op_type) != "CONCATENATION":
            return False
        concat_cf_spec = resolve_concat_axis_for_channel_first_fn(consumer_op)
        if concat_cf_spec is None:
            return False
        consumer_output_name = str(consumer_op.outputs[0]) if len(consumer_op.outputs) == 1 else ""
        if consumer_output_name == "":
            return False
        consumer_output_tensor = model_ir.tensors.get(consumer_output_name, None)
        if consumer_output_tensor is None:
            return False
        consumer_output_rank = len(list(consumer_output_tensor.shape))
        consumer_output_layout = normalize_logical_layout(consumer_output_tensor.logical_layout)
        if consumer_output_layout in {
            LOGICAL_LAYOUT_UNKNOWN,
            channel_first_logical_layout(consumer_output_rank),
        }:
            continue
        return False
    return True

def _reshape_codegen_is_plain_data_only_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    infer_effective_rank4_runtime_layout_fn: Callable[[str], Optional[str]],
    reshape_preserves_channel_last_sequence_fn: Callable[[Sequence[int], Sequence[int], str], Optional[List[int]]],
    reshape_prefers_feature_last_for_adjx_batch_matmul_fn: Callable[[str, str], Optional[Tuple[List[int], List[int]]]],
) -> bool:
    if str(op.op_type) != "RESHAPE" or len(op.inputs) == 0 or len(op.outputs) == 0:
        return False
    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if input_tensor is None or output_tensor is None:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    input_layout = str(input_tensor.logical_layout)
    output_layout = str(output_tensor.logical_layout)
    reshape_is_lowered_onnx_flatten = "onnxFlattenAxis" in op.options
    if (
        not reshape_is_lowered_onnx_flatten
        and len(input_shape) == 4
        and len(output_shape) in {2, 3}
    ):
        effective_layout = infer_effective_rank4_runtime_layout_fn(str(op.inputs[0]))
        if effective_layout is not None:
            input_layout = effective_layout
    reshape_plain_singleton_axis_drop = _reshape_is_plain_singleton_axis_drop(
        input_shape,
        output_shape,
    )
    if reshape_is_lowered_onnx_flatten or reshape_plain_singleton_axis_drop:
        return True
    reshape_special_plan = _reshape_special_layout_plan(
        input_shape=input_shape,
        output_shape=output_shape,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    reshape_pre_perm = reshape_preserves_channel_last_sequence_fn(
        input_shape,
        output_shape,
        input_layout,
    )
    if reshape_special_plan is not None and reshape_special_plan.get("pre_perm", None) is not None:
        reshape_pre_perm = list(reshape_special_plan["pre_perm"])
    reshape_feature_last_target = reshape_prefers_feature_last_for_adjx_batch_matmul_fn(
        str(op.inputs[0]),
        str(op.outputs[0]),
    )
    if reshape_feature_last_target is not None:
        reshape_pre_perm = list(reshape_feature_last_target[0])
    reshape_channel_first_alias_shape = (
        len(input_shape) == 2
        and len(output_shape) == 4
        and is_channel_last_logical_layout(normalize_logical_layout(output_tensor.logical_layout))
        and all(int(dim) == 1 for dim in list(output_shape[1:-1]))
        and _shape_lists_equal_relaxed(
            input_shape,
            [int(output_shape[0]), int(output_shape[-1])],
        )
    )
    return not bool(
        reshape_special_plan is not None
        or reshape_pre_perm is not None
        or reshape_feature_last_target is not None
        or reshape_channel_first_alias_shape
    )

def _tensor_exact_static_shape_list_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    if (
        tensor.shape_signature is not None
        and len(list(tensor.shape_signature)) == len(list(tensor.shape))
    ):
        signature = [int(v) for v in list(tensor.shape_signature)]
        if all(int(v) > 0 for v in signature):
            return signature
    return None

def _static_sequence_length_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[int]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    if len(list(tensor.shape)) >= 1 and int(tensor.shape[0]) > 0:
        return int(tensor.shape[0])
    if tensor.shape_signature is not None and len(list(tensor.shape_signature)) >= 1 and int(tensor.shape_signature[0]) > 0:
        return int(tensor.shape_signature[0])
    return None

def _is_identity_nms_postprocess_gather_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_expr_aliases: Dict[str, str],
    producer_index: Dict[str, int],
    scalar_literal_expr_fn: Callable[[str], Optional[str]],
    params_name: str,
    indices_name: str,
) -> bool:
    params_alias = tensor_expr_aliases.get(str(params_name), "")
    if not str(params_alias).startswith("_nms_selected_indices_valid_"):
        return False
    indices_producer_index = producer_index.get(str(indices_name), None)
    if indices_producer_index is None:
        return False
    indices_producer = model_ir.operators[int(indices_producer_index)]
    if str(indices_producer.op_type) != "RANGE" or len(indices_producer.inputs) < 3:
        return False
    start_literal = scalar_literal_expr_fn(str(indices_producer.inputs[0]))
    delta_literal = scalar_literal_expr_fn(str(indices_producer.inputs[2]))
    return start_literal == "0" and delta_literal == "1"

def _range_only_feeds_identity_nms_postprocess_gathers_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    is_identity_nms_postprocess_gather_fn: Callable[[str, str], bool],
    output_name: str,
) -> bool:
    consumers = consumer_index.get(str(output_name), [])
    if len(consumers) == 0:
        return False
    for consumer_idx in consumers:
        consumer_op = model_ir.operators[int(consumer_idx)]
        if str(consumer_op.op_type) != "GATHER" or len(consumer_op.inputs) < 2:
            return False
        if str(consumer_op.inputs[1]) != str(output_name):
            return False
        if not is_identity_nms_postprocess_gather_fn(str(consumer_op.inputs[0]), str(output_name)):
            return False
    return True

def _conv2d_output_spatial_shape_for_codegen(
    *,
    input_hw: Sequence[int],
    kernel_hw: Sequence[int],
    stride_hw: Sequence[int],
    dilation_hw: Sequence[int],
    padding_mode: str,
) -> Optional[List[int]]:
    input_items = [int(v) for v in list(input_hw)]
    kernel_items = [int(v) for v in list(kernel_hw)]
    stride_items = [max(1, int(v)) for v in list(stride_hw)]
    dilation_items = [max(1, int(v)) for v in list(dilation_hw)]
    if len(input_items) != 2 or len(kernel_items) != 2:
        return None
    if len(stride_items) != 2 or len(dilation_items) != 2:
        return None
    if any(int(v) <= 0 for v in input_items + kernel_items):
        return None
    padding_key = str(padding_mode).upper()
    output_hw: List[int] = []
    for input_dim, kernel_dim, stride_dim, dilation_dim in zip(
        input_items,
        kernel_items,
        stride_items,
        dilation_items,
    ):
        effective_kernel = (int(kernel_dim) - 1) * int(dilation_dim) + 1
        if padding_key == "SAME":
            output_dim = int(math.ceil(float(input_dim) / float(stride_dim)))
        elif padding_key == "VALID":
            output_dim = int(math.floor((float(input_dim) - float(effective_kernel)) / float(stride_dim))) + 1
        else:
            return None
        if int(output_dim) <= 0:
            return None
        output_hw.append(int(output_dim))
    return output_hw

def _conv3d_output_spatial_shape_for_codegen(
    *,
    input_dhw: Sequence[int],
    kernel_dhw: Sequence[int],
    stride_dhw: Sequence[int],
    dilation_dhw: Sequence[int],
    padding_mode: str,
) -> Optional[List[int]]:
    input_items = [int(v) for v in list(input_dhw)]
    kernel_items = [int(v) for v in list(kernel_dhw)]
    stride_items = [max(1, int(v)) for v in list(stride_dhw)]
    dilation_items = [max(1, int(v)) for v in list(dilation_dhw)]
    if len(input_items) != 3 or len(kernel_items) != 3:
        return None
    if len(stride_items) != 3 or len(dilation_items) != 3:
        return None
    if any(int(v) <= 0 for v in input_items + kernel_items):
        return None
    padding_key = str(padding_mode).upper()
    output_dhw: List[int] = []
    for input_dim, kernel_dim, stride_dim, dilation_dim in zip(
        input_items,
        kernel_items,
        stride_items,
        dilation_items,
    ):
        effective_kernel = (int(kernel_dim) - 1) * int(dilation_dim) + 1
        if padding_key == "SAME":
            output_dim = int(math.ceil(float(input_dim) / float(stride_dim)))
        elif padding_key == "VALID":
            output_dim = int(math.floor((float(input_dim) - float(effective_kernel)) / float(stride_dim))) + 1
        else:
            return None
        if int(output_dim) <= 0:
            return None
        output_dhw.append(int(output_dim))
    return output_dhw

def _conv3d_transpose_output_spatial_shape_for_codegen(
    *,
    input_dhw: Sequence[int],
    kernel_dhw: Sequence[int],
    stride_dhw: Sequence[int],
    dilation_dhw: Sequence[int],
    padding_mode: str,
) -> Optional[List[int]]:
    input_items = [int(v) for v in list(input_dhw)]
    kernel_items = [int(v) for v in list(kernel_dhw)]
    stride_items = [max(1, int(v)) for v in list(stride_dhw)]
    dilation_items = [max(1, int(v)) for v in list(dilation_dhw)]
    if len(input_items) != 3 or len(kernel_items) != 3 or len(stride_items) != 3 or len(dilation_items) != 3:
        return None
    if any(int(v) <= 0 for v in input_items + kernel_items):
        return None
    padding_key = str(padding_mode).upper()
    output_dhw: List[int] = []
    for input_dim, kernel_dim, stride_dim, dilation_dim in zip(
        input_items,
        kernel_items,
        stride_items,
        dilation_items,
    ):
        effective_kernel = (int(kernel_dim) - 1) * int(dilation_dim) + 1
        if padding_key == "SAME":
            output_dim = int(input_dim) * int(stride_dim)
        elif padding_key == "VALID":
            output_dim = (int(input_dim) - 1) * int(stride_dim) + int(effective_kernel)
        else:
            return None
        if int(output_dim) <= 0:
            return None
        output_dhw.append(int(output_dim))
    return output_dhw

def _conv2d_same_pad_arg_for_codegen(
    *,
    input_hw: Sequence[int],
    kernel_hw: Sequence[int],
    stride_hw: Sequence[int],
    dilation_hw: Sequence[int],
) -> Optional[Tuple[int, int]]:
    input_items = [int(v) for v in list(input_hw)]
    kernel_items = [int(v) for v in list(kernel_hw)]
    stride_items = [max(1, int(v)) for v in list(stride_hw)]
    dilation_items = [max(1, int(v)) for v in list(dilation_hw)]
    if len(input_items) != 2 or len(kernel_items) != 2:
        return None
    if len(stride_items) != 2 or len(dilation_items) != 2:
        return None
    if any(int(v) <= 0 for v in input_items + kernel_items):
        return None
    pad_values: List[int] = []
    for input_dim, kernel_dim, stride_dim, dilation_dim in zip(
        input_items,
        kernel_items,
        stride_items,
        dilation_items,
    ):
        output_dim = int(math.ceil(float(input_dim) / float(stride_dim)))
        effective_kernel = (int(kernel_dim) - 1) * int(dilation_dim) + 1
        total_pad = max(0, (int(output_dim) - 1) * int(stride_dim) + int(effective_kernel) - int(input_dim))
        if total_pad % 2 != 0:
            return None
        pad_values.append(int(total_pad // 2))
    return int(pad_values[0]), int(pad_values[1])

def _conv2d_same_pad_padding_arg_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_pre_permute: Optional[Sequence[int]] = None,
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
) -> Optional[List[int]]:
    if str((options or {}).get("padding", "SAME")).upper() != "SAME":
        return None
    if input_shape is None or output_shape is None or weight_shape is None:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
        return None
    if input_pre_permute is not None:
        perm = [int(v) for v in list(input_pre_permute)]
        if len(perm) != 4:
            return None
        if not is_channel_first_logical_layout(normalized_input_layout):
            in_shape = [int(in_shape[idx]) for idx in perm]
        should_permute_output_shape = normalized_output_layout == "NHWC"
        if (
            not should_permute_output_shape
            and int(kernel_shape[0]) > 0
            and int(out_shape[1]) != int(kernel_shape[0])
            and int(out_shape[-1]) == int(kernel_shape[0])
        ):
            should_permute_output_shape = True
        if should_permute_output_shape:
            out_shape = [int(out_shape[idx]) for idx in perm]
    else:
        if is_channel_last_logical_layout(normalized_input_layout):
            perm = _perm_cl_to_cf(4)
            if perm is not None:
                in_shape = [int(in_shape[idx]) for idx in perm]
        if is_channel_last_logical_layout(normalized_output_layout):
            perm = _perm_cl_to_cf(4)
            if perm is not None:
                out_shape = [int(out_shape[idx]) for idx in perm]
    stride_hw = [
        max(1, int((options or {}).get("strideH", 1))),
        max(1, int((options or {}).get("strideW", 1))),
    ]
    dilation_hw = [
        max(1, int((options or {}).get("dilationHFactor", 1))),
        max(1, int((options or {}).get("dilationWFactor", 1))),
    ]
    input_hw = [int(in_shape[2]), int(in_shape[3])]
    output_hw = [int(out_shape[2]), int(out_shape[3])]
    if output_hw[0] <= 0:
        output_hw[0] = max(1, int(math.ceil(float(input_hw[0]) / float(stride_hw[0]))))
    if output_hw[1] <= 0:
        output_hw[1] = max(1, int(math.ceil(float(input_hw[1]) / float(stride_hw[1]))))
    for idx in range(2):
        if int(input_hw[idx]) <= 0 and int(output_hw[idx]) > 0:
            input_hw[idx] = max(1, int(output_hw[idx]) * int(stride_hw[idx]))
    kernel_hw_candidates: List[List[int]] = []
    expected_in_channels = int(in_shape[1]) if len(in_shape) == 4 else -1
    if expected_in_channels > 0:
        if (
            int(kernel_shape[1]) == expected_in_channels
            or (int(kernel_shape[1]) == 1 and int(kernel_shape[0]) == expected_in_channels)
        ):
            kernel_hw_candidates.append([2, 3])
        if (
            int(kernel_shape[3]) == expected_in_channels
            or (int(kernel_shape[0]) == 1 and int(kernel_shape[3]) == expected_in_channels)
        ):
            kernel_hw_candidates.append([1, 2])
    if len(kernel_hw_candidates) == 0:
        kernel_hw_candidates.append([2, 3])
        if [1, 2] not in kernel_hw_candidates:
            kernel_hw_candidates.append([1, 2])
    best_pad_totals: Optional[Tuple[int, int]] = None
    effective_kernel: Optional[List[int]] = None
    for kernel_hw_indices in kernel_hw_candidates:
        candidate_kernel = [
            (int(kernel_shape[int(kernel_hw_indices[idx])]) - 1) * int(dilation_hw[idx]) + 1
            for idx in range(2)
        ]
        candidate_pad_h_total = max(
            (int(output_hw[0]) - 1) * int(stride_hw[0]) + int(candidate_kernel[0]) - int(input_hw[0]),
            0,
        )
        candidate_pad_w_total = max(
            (int(output_hw[1]) - 1) * int(stride_hw[1]) + int(candidate_kernel[1]) - int(input_hw[1]),
            0,
        )
        candidate_pad_totals = (int(candidate_pad_h_total), int(candidate_pad_w_total))
        if best_pad_totals is None or candidate_pad_totals < best_pad_totals:
            best_pad_totals = candidate_pad_totals
            effective_kernel = candidate_kernel
    if effective_kernel is None or best_pad_totals is None:
        return None
    pad_h_total, pad_w_total = best_pad_totals
    if pad_h_total == 0 and pad_w_total == 0:
        return None
    pad_top = int(pad_h_total // 2)
    pad_bottom = int(pad_h_total - pad_top)
    pad_left = int(pad_w_total // 2)
    pad_right = int(pad_w_total - pad_left)
    return [pad_left, pad_right, pad_top, pad_bottom]

def _reshape_preserves_channel_last_sequence_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    input_layout: Optional[str],
) -> Optional[List[int]]:
    if input_shape is None or output_shape is None:
        return None
    src = [int(v) for v in list(input_shape)]
    dst = [int(v) for v in list(output_shape)]
    layout = str(input_layout or "").upper()
    if layout == "NCHW" and len(src) == 4 and len(dst) == 3:
        flattened_spatial = int(src[2]) * int(src[3])
        sequence_extent_matches = (
            dst[1] == -1
            or (dst[2] > 0 and flattened_spatial * max(1, int(src[1]) // int(dst[2])) == dst[1])
            or flattened_spatial == dst[1]
        )
        if (
            src[0] == dst[0]
            and dst[2] > 0
            and int(src[1]) % int(dst[2]) == 0
            and src[2] > 0
            and src[3] > 0
            and sequence_extent_matches
        ):
            return [0, 2, 3, 1]
    if layout == "NCDHW" and len(src) == 5 and len(dst) == 3:
        spatial = src[2] * src[3] * src[4]
        sequence_extent_matches = (
            dst[1] == -1
            or (dst[2] > 0 and spatial * max(1, int(src[1]) // int(dst[2])) == dst[1])
            or spatial == dst[1]
        )
        if (
            src[0] == dst[0]
            and dst[2] > 0
            and int(src[1]) % int(dst[2]) == 0
            and sequence_extent_matches
        ):
            return [0, 2, 3, 4, 1]
    if layout == "NCW" and len(src) == 3 and len(dst) == 3:
        sequence_extent_matches = (
            dst[1] == -1
            or (dst[2] > 0 and int(src[2]) * max(1, int(src[1]) // int(dst[2])) == dst[1])
            or src[2] == dst[1]
        )
        if (
            src[0] == dst[0]
            and dst[2] > 0
            and int(src[1]) % int(dst[2]) == 0
            and sequence_extent_matches
        ):
            return [0, 2, 1]
    return None

def _reshape_prefers_feature_last_for_adjx_batch_matmul_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    input_tensor_name: str,
    output_name: str,
) -> Optional[Tuple[List[int], List[int]]]:
    input_tensor = model_ir.tensors.get(str(input_tensor_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if input_tensor is None or output_tensor is None:
        return None
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    preferred_pre_perm: Optional[List[int]] = None
    preferred_shape: Optional[List[int]] = None
    if (
        len(input_shape) == 3
        and len(output_shape) == 3
        and input_layout == "NCW"
        and output_layout == "NCW"
        and input_shape[0] == 1
        and input_shape[1] > 1
        and input_shape[2] > 1
        and output_shape[0] == input_shape[2]
        and output_shape[1] == 1
        and output_shape[2] == input_shape[1]
    ):
        preferred_pre_perm = [0, 1, 2]
        preferred_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[1])]
    elif (
        len(input_shape) == 4
        and len(output_shape) == 3
        and input_layout == "NCHW"
        and output_layout == "NCW"
        and output_shape[0] == input_shape[0]
        and int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
    ):
        preferred_pre_perm = [0, 3, 1, 2]
        preferred_shape = [
            int(input_shape[0]),
            int(input_shape[3]),
            int(input_shape[1]) * int(input_shape[2]),
        ]
    if preferred_pre_perm is not None and preferred_shape is not None:
        pending_outputs: List[str] = [str(output_name)]
        visited_outputs: Set[str] = set()
        while pending_outputs:
            current_name = pending_outputs.pop()
            if current_name in visited_outputs:
                continue
            visited_outputs.add(current_name)
            for consumer_idx in consumer_index.get(current_name, []):
                consumer_op = model_ir.operators[int(consumer_idx)]
                consumer_type = str(consumer_op.op_type)
                if consumer_type == "BATCH_MATMUL":
                    if len(consumer_op.inputs) < 2 or str(consumer_op.inputs[0]) != current_name:
                        continue
                    if not bool(consumer_op.options.get("adjX", False)):
                        continue
                    rhs_tensor = model_ir.tensors.get(str(consumer_op.inputs[1]), None)
                    if rhs_tensor is None or len(list(rhs_tensor.shape)) < 2:
                        continue
                    rhs_contract = int(list(rhs_tensor.shape)[-2])
                    expected_contract = (
                        int(input_shape[1]) if len(input_shape) == 3 else int(preferred_shape[1])
                    )
                    if rhs_contract != expected_contract:
                        continue
                    return (list(preferred_pre_perm), list(preferred_shape))
                if consumer_type in {
                    "ABS",
                    "ATAN",
                    "CAST",
                    "ELU",
                    "ERF",
                    "EXP",
                    "GELU",
                    "IDENTITY",
                    "LEAKY_RELU",
                    "LOG",
                    "LOGISTIC",
                    "NEG",
                    "RELU",
                    "RELU6",
                    "RELU_0_TO_1",
                    "RELU_N1_TO_1",
                    "SIGN",
                    "SIN",
                    "SQRT",
                    "SQUARE",
                    "TANH",
                }:
                    if len(consumer_op.outputs) == 1 and len(consumer_op.inputs) >= 1 and str(consumer_op.inputs[0]) == current_name:
                        pending_outputs.append(str(consumer_op.outputs[0]))
        return None
    return None

def _matmul_broadcast_shape_for_codegen(
    *,
    lhs_batch: Sequence[int],
    rhs_batch: Sequence[int],
) -> Optional[List[int]]:
    lhs_items = [int(v) for v in list(lhs_batch)]
    rhs_items = [int(v) for v in list(rhs_batch)]
    result: List[int] = []
    for lhs_dim, rhs_dim in zip(reversed(lhs_items), reversed(rhs_items)):
        if int(lhs_dim) == int(rhs_dim):
            result.append(int(lhs_dim))
        elif int(lhs_dim) == 1:
            result.append(int(rhs_dim))
        elif int(rhs_dim) == 1:
            result.append(int(lhs_dim))
        else:
            return None
    if len(lhs_items) > len(rhs_items):
        result.extend(reversed(lhs_items[: len(lhs_items) - len(rhs_items)]))
    elif len(rhs_items) > len(lhs_items):
        result.extend(reversed(rhs_items[: len(rhs_items) - len(lhs_items)]))
    return list(reversed(result))

def _infer_batch_matmul_shape_for_codegen(
    *,
    lhs_shape: Optional[Sequence[int]],
    rhs_shape: Optional[Sequence[int]],
    adj_x: bool,
    adj_y: bool,
) -> Optional[List[int]]:
    if lhs_shape is None or rhs_shape is None:
        return None
    lhs_items = [int(v) for v in list(lhs_shape)]
    rhs_items = [int(v) for v in list(rhs_shape)]
    if len(lhs_items) == 0 or len(rhs_items) == 0:
        return None
    if len(lhs_items) == 1:
        lhs_items = [1, int(lhs_items[0])]
    if len(rhs_items) == 1:
        rhs_items = [int(rhs_items[0]), 1]
    if len(lhs_items) < 2 or len(rhs_items) < 2:
        return None
    lhs_m = int(lhs_items[-1 if adj_x else -2])
    lhs_k = int(lhs_items[-2 if adj_x else -1])
    rhs_k = int(rhs_items[-1 if adj_y else -2])
    rhs_n = int(rhs_items[-2 if adj_y else -1])
    if int(lhs_k) != int(rhs_k):
        return None
    batch_shape = _matmul_broadcast_shape_for_codegen(
        lhs_batch=lhs_items[:-2],
        rhs_batch=rhs_items[:-2],
    )
    if batch_shape is None:
        return None
    return list(batch_shape) + [int(lhs_m), int(rhs_n)]

def _infer_reduction_shape_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    axes: Optional[Sequence[int]],
    keepdims: bool,
) -> Optional[List[int]]:
    if input_shape is None:
        return None
    dims = [int(v) for v in list(input_shape)]
    if axes is None:
        return [1 for _ in dims] if keepdims else []
    normalized_axes = sorted({int(v) for v in list(axes)})
    if keepdims:
        return [1 if idx in normalized_axes else int(dim) for idx, dim in enumerate(dims)]
    return [int(dim) for idx, dim in enumerate(dims) if idx not in normalized_axes]

def _infer_gather_nd_shape_for_codegen(
    *,
    model_ir: ModelIR,
    params_shape: Optional[Sequence[int]],
    indices_tensor_name: str,
) -> Optional[List[int]]:
    if params_shape is None:
        return None
    indices_tensor = model_ir.tensors.get(str(indices_tensor_name), None)
    if indices_tensor is None:
        return None
    indices_shape = [int(v) for v in list(indices_tensor.shape)]
    if len(indices_shape) == 0:
        return None
    index_depth = int(indices_shape[-1])
    params_items = [int(v) for v in list(params_shape)]
    if index_depth > len(params_items):
        return None
    return indices_shape[:-1] + params_items[index_depth:]

def _emit_maybe_aligned_expr_for_codegen(
    *,
    runtime_imports: Set[str],
    output_name: str,
    expr: str,
    inferred_shape: Optional[Sequence[int]],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    target_shape_literal_fn: Callable[[str], str],
) -> str:
    output_shape = tensor_shape_list_fn(output_name)
    if _shape_lists_equal(inferred_shape, output_shape):
        return expr
    runtime_imports.add("_align_tensor_to_target_shape")
    return f"_align_tensor_to_target_shape({expr}, {target_shape_literal_fn(output_name)})"

def _emit_module_output_expr_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_imports: Set[str],
    output_name: str,
    expr: str,
    raw_output_layout: str,
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    target_shape_literal_fn: Callable[[str], str],
) -> str:
    output_tensor = model_ir.tensors.get(str(output_name), None)
    output_layout = (
        normalize_logical_layout(output_tensor.logical_layout)
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    normalized_raw_layout = normalize_logical_layout(raw_output_layout)
    if (
        output_layout != LOGICAL_LAYOUT_UNKNOWN
        and normalized_raw_layout != LOGICAL_LAYOUT_UNKNOWN
        and output_layout != normalized_raw_layout
    ):
        output_shape = tensor_shape_list_fn(output_name)
        rank = (
            len(list(output_shape))
            if output_shape is not None
            else len(list(output_tensor.shape)) if output_tensor is not None else 0
        )
        if rank in {3, 4, 5}:
            perm = logical_layout_permutation(
                source_layout=normalized_raw_layout,
                target_layout=output_layout,
            )
            if perm is not None:
                expr = f"{expr}.permute({', '.join(str(int(v)) for v in perm)}).contiguous()"
    runtime_imports.add("_align_tensor_to_target_shape")
    return f"_align_tensor_to_target_shape({expr}, {target_shape_literal_fn(output_name)})"

def _is_constant_tensor_name_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> bool:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    return tensor is not None and isinstance(tensor.data, np.ndarray)

def _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> bool:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.integer):
        return False
    return not bool(np.any(arr.reshape(-1) == 0))

def _static_int_tensor_values_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    tensor_name: str,
    _visited: Optional[Set[str]] = None,
) -> Optional[List[int]]:
    visited = set() if _visited is None else _visited
    current_name = str(tensor_name)
    if current_name in visited:
        return None
    visited.add(current_name)

    direct_values = _constant_int_list(model_ir.tensors.get(current_name, None))
    if direct_values is not None:
        return [int(v) for v in list(direct_values)]

    producer_idx = producer_index.get(current_name, None)
    if producer_idx is None:
        return None

    producer = model_ir.operators[int(producer_idx)]
    op_type = str(producer.op_type)

    def _static_shape_values(input_name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(input_name), None)
        if tensor is None:
            return None
        shape_values = (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            and len(list(tensor.shape_signature)) == len(list(tensor.shape))
            else [int(v) for v in list(tensor.shape)]
        )
        if any(int(v) <= 0 for v in shape_values):
            return None
        return shape_values

    def _scalar_or_vector_int_values(input_name: str) -> Optional[List[int]]:
        return _static_int_tensor_values_for_codegen(
            model_ir=model_ir,
            producer_index=producer_index,
            tensor_name=str(input_name),
            _visited=set(visited),
        )

    if op_type == "SHAPE" and len(producer.inputs) >= 1:
        return _static_shape_values(str(producer.inputs[0]))

    if op_type in {"CAST", "EXPAND_DIMS", "IDENTITY", "RESHAPE", "SQUEEZE"} and len(producer.inputs) >= 1:
        return _scalar_or_vector_int_values(str(producer.inputs[0]))

    if op_type == "GATHER" and len(producer.inputs) >= 2:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        gather_indices = _scalar_or_vector_int_values(str(producer.inputs[1]))
        axis = int(producer.options.get("axis", 0))
        batch_dims = int(producer.options.get("batchDims", 0))
        if input_values is None or gather_indices is None or batch_dims != 0:
            return None
        if axis < 0:
            axis += 1
        if axis != 0:
            return None
        gathered: List[int] = []
        for raw_index in gather_indices:
            index = int(raw_index)
            if index < 0:
                index += len(input_values)
            if index < 0 or index >= len(input_values):
                return None
            gathered.append(int(input_values[index]))
        return gathered

    if op_type == "GATHER_ND" and len(producer.inputs) >= 2:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        gather_indices = _constant_int_list(model_ir.tensors.get(str(producer.inputs[1]), None))
        if input_values is None or gather_indices is None:
            return None
        if len(gather_indices) == 1:
            index = int(gather_indices[0])
            if index < 0:
                index += len(input_values)
            if index < 0 or index >= len(input_values):
                return None
            return [int(input_values[index])]
        return None

    if op_type == "SLICE" and len(producer.inputs) >= 3:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        begin_values = _scalar_or_vector_int_values(str(producer.inputs[1]))
        size_values = _scalar_or_vector_int_values(str(producer.inputs[2]))
        if (
            input_values is None
            or begin_values is None
            or size_values is None
            or len(begin_values) != 1
            or len(size_values) != 1
        ):
            return None
        start = int(begin_values[0])
        if start < 0:
            start += len(input_values)
        size = int(size_values[0])
        stop = None if size < 0 else start + size
        return [int(v) for v in input_values[slice(start, stop)]]

    if op_type == "STRIDED_SLICE" and len(producer.inputs) >= 4:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        begin_values = _scalar_or_vector_int_values(str(producer.inputs[1]))
        end_values = _scalar_or_vector_int_values(str(producer.inputs[2]))
        stride_values = _scalar_or_vector_int_values(str(producer.inputs[3]))
        if (
            input_values is None
            or begin_values is None
            or end_values is None
            or stride_values is None
            or len(begin_values) != 1
            or len(end_values) != 1
            or len(stride_values) != 1
        ):
            return None
        begin_mask = int(producer.options.get("beginMask", 0))
        end_mask = int(producer.options.get("endMask", 0))
        start = None if (begin_mask & 1) else int(begin_values[0])
        stop = None if (end_mask & 1) else int(end_values[0])
        step = int(stride_values[0])
        if step == 0:
            return None
        return [int(v) for v in input_values[slice(start, stop, step)]]

    if op_type in {"CONCATENATION", "PACK"}:
        output_values: List[int] = []
        for input_name in producer.inputs:
            input_values = _scalar_or_vector_int_values(str(input_name))
            if input_values is None:
                return None
            output_values.extend(int(v) for v in input_values)
        return output_values

    if op_type in {"MAXIMUM", "MINIMUM"} and len(producer.inputs) >= 2:
        lhs_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        rhs_values = _scalar_or_vector_int_values(str(producer.inputs[1]))
        if lhs_values is None or rhs_values is None:
            return None
        lhs_array = np.asarray(lhs_values, dtype=np.int64)
        rhs_array = np.asarray(rhs_values, dtype=np.int64)
        try:
            output_array = (
                np.maximum(lhs_array, rhs_array)
                if op_type == "MAXIMUM"
                else np.minimum(lhs_array, rhs_array)
            )
        except ValueError:
            return None
        return [int(v) for v in output_array.reshape(-1).tolist()]

    return None

def _reshape_shape_tensor_uses_runtime_dims_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    tensor_name: str,
    _visited: Optional[Set[str]] = None,
) -> bool:
    visited = set() if _visited is None else _visited
    current_name = str(tensor_name)
    if current_name in visited:
        return False
    visited.add(current_name)

    producer_idx = producer_index.get(current_name, None)
    if producer_idx is None:
        return _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
            model_ir=model_ir,
            tensor_name=current_name,
        )

    producer = model_ir.operators[int(producer_idx)]
    op_type = str(producer.op_type)
    if op_type == "SHAPE":
        return True
    if op_type in {"CAST", "EXPAND_DIMS", "IDENTITY", "RESHAPE", "SQUEEZE"}:
        return (
            len(producer.inputs) >= 1
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[0]),
                _visited=set(visited),
            )
        )
    if op_type in {"GATHER", "GATHER_ND", "SLICE", "STRIDED_SLICE"}:
        return (
            len(producer.inputs) >= 1
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[0]),
                _visited=set(visited),
            )
        )
    if op_type == "SPLIT":
        split_data_input_index = 1 if len(producer.inputs) >= 2 else 0
        return (
            len(producer.inputs) > split_data_input_index
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[split_data_input_index]),
                _visited=set(visited),
            )
        )
    if op_type == "UNPACK":
        return (
            len(producer.inputs) >= 1
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[0]),
                _visited=set(visited),
            )
        )
    if op_type in {"CONCATENATION", "PACK"}:
        saw_runtime_dims = False
        for input_name in producer.inputs:
            if _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(input_name),
                _visited=set(visited),
            ):
                saw_runtime_dims = True
                continue
            if _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
                model_ir=model_ir,
                tensor_name=str(input_name),
            ):
                continue
            return False
        return saw_runtime_dims
    return False

def _should_skip_align_for_shape_preserving_unary_for_codegen(
    *,
    model_ir: ModelIR,
    input_name: str,
    output_name: str,
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
) -> bool:
    input_tensor = model_ir.tensors.get(str(input_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if input_tensor is None or output_tensor is None:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if input_layout != output_layout:
        return False
    input_shape = tensor_shape_list_fn(input_name)
    output_shape = tensor_shape_list_fn(output_name)
    if input_shape is None or output_shape is None:
        return False
    if _shape_lists_equal(input_shape, output_shape):
        return True
    if len(input_shape) != len(output_shape):
        return False
    try:
        return int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
    except Exception:
        return False

def _next_unique_attr_name_for_codegen(
    *,
    base_name: str,
    module_attr_counts: Dict[str, int],
    affine_layer_norm_specs: Dict[int, Dict[str, Any]],
    op_module_attr_names: Dict[int, str],
) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(base_name)).strip("_").lower()
    if len(normalized) == 0:
        normalized = "generated_module"
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    candidate = normalized
    suffix = 1
    existing_names = {
        *module_attr_counts.keys(),
        *(str(spec.get("attr_name")) for spec in affine_layer_norm_specs.values()),
        *(str(value) for value in op_module_attr_names.values()),
    }
    while candidate in existing_names:
        candidate = f"{normalized}_{suffix}"
        suffix += 1
    module_attr_counts[candidate] = 1
    return candidate

def _canonical_codegen_name_for_codegen(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")

def _match_affine_layer_norm_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    is_constant_tensor_name_fn: Callable[[str], bool],
    canonical_codegen_name_fn: Callable[[str], str],
    next_unique_attr_name_fn: Callable[[str], str],
    op_index: int,
    op: OperatorIR,
) -> Optional[Dict[str, Any]]:
    if str(op.op_type) != "ADD" or len(op.inputs) < 2 or len(op.outputs) != 1:
        return None
    output_name = str(op.outputs[0])
    canonical_output_name = canonical_codegen_name_fn(output_name)
    if "fakelayernorm" not in canonical_output_name or not canonical_output_name.endswith("add"):
        return None
    beta_input_name = ""
    mul_output_name = ""
    for input_name in op.inputs[:2]:
        input_tensor_name = str(input_name)
        canonical_input_name = canonical_codegen_name_fn(input_tensor_name)
        if is_constant_tensor_name_fn(input_tensor_name) and "fakelayernorm_beta" in canonical_input_name:
            beta_input_name = input_tensor_name
        else:
            mul_output_name = input_tensor_name
    if beta_input_name == "" or mul_output_name == "":
        return None
    mul_op_index = producer_index.get(str(mul_output_name), None)
    if mul_op_index is None:
        return None
    mul_op = model_ir.operators[int(mul_op_index)]
    if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) < 2 or len(mul_op.outputs) != 1:
        return None
    if str(mul_op.outputs[0]) != mul_output_name:
        return None
    gamma_input_name = ""
    input_name = ""
    for mul_input_name in mul_op.inputs[:2]:
        candidate_name = str(mul_input_name)
        canonical_candidate_name = canonical_codegen_name_fn(candidate_name)
        if is_constant_tensor_name_fn(candidate_name) and "fakelayernorm_gamma" in canonical_candidate_name:
            gamma_input_name = candidate_name
        else:
            input_name = candidate_name
    if gamma_input_name == "" or input_name == "":
        return None
    gamma_tensor = model_ir.tensors.get(str(gamma_input_name), None)
    beta_tensor = model_ir.tensors.get(str(beta_input_name), None)
    if gamma_tensor is None or beta_tensor is None:
        return None
    attr_stem = re.sub(r"(?i)(?:[/_])?FakeLayerNorm(?:[/_])add$", "", output_name)
    attr_stem = re.sub(r"^bert[/_]", "", attr_stem, flags=re.IGNORECASE)
    attr_name = next_unique_attr_name_fn(f"{attr_stem}_layer_norm")
    return {
        "attr_name": attr_name,
        "input_name": str(input_name),
        "output_name": output_name,
        "gamma_name": str(gamma_input_name),
        "beta_name": str(beta_input_name),
        "gamma_shape": [int(v) for v in list(gamma_tensor.shape)],
        "gamma_dtype": str(gamma_tensor.dtype).upper(),
        "mul_op_index": int(mul_op_index),
    }

def _match_swish_activation_pattern_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    tensor_name: str,
    consumer_indices: Sequence[int],
) -> Optional[Tuple[str, Set[int]]]:
    consumer_index_list = [int(idx) for idx in consumer_indices]
    if len(consumer_index_list) != 2:
        return None
    logistic_idx: Optional[int] = None
    mul_idx: Optional[int] = None
    logistic_output_name: Optional[str] = None
    for consumer_idx in consumer_index_list:
        consumer_op = model_ir.operators[int(consumer_idx)]
        consumer_type = str(consumer_op.op_type)
        if (
            consumer_type == "LOGISTIC"
            and len(consumer_op.inputs) == 1
            and len(consumer_op.outputs) == 1
            and str(consumer_op.inputs[0]) == str(tensor_name)
        ):
            logistic_idx = int(consumer_idx)
            logistic_output_name = str(consumer_op.outputs[0])
            continue
        if (
            consumer_type == "MUL"
            and len(consumer_op.inputs) == 2
            and len(consumer_op.outputs) == 1
            and str(tensor_name) in {str(name) for name in list(consumer_op.inputs)}
        ):
            mul_idx = int(consumer_idx)
    if logistic_idx is None or mul_idx is None or logistic_output_name is None:
        return None
    mul_op = model_ir.operators[int(mul_idx)]
    mul_input_names = [str(name) for name in list(mul_op.inputs)]
    if logistic_output_name not in mul_input_names:
        return None
    if set(mul_input_names) != {str(tensor_name), str(logistic_output_name)}:
        return None
    if consumer_index.get(str(logistic_output_name), []) != [int(mul_idx)]:
        return None
    return (str(mul_op.outputs[0]), {int(logistic_idx), int(mul_idx)})

def _topk_codegen_layout_bridge_for_codegen(
    *,
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    input_name: str,
    value_output_name: str,
    index_output_name: Optional[str],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    input_shape = tensor_shape_list_fn(str(input_name))
    value_shape = tensor_shape_list_fn(str(value_output_name))
    index_shape = (
        tensor_shape_list_fn(str(index_output_name))
        if index_output_name is not None and str(index_output_name) != ""
        else None
    )
    if input_shape is None or value_shape is None:
        return None, None
    rank = len(input_shape)
    if rank not in {3, 4, 5} or len(value_shape) != rank:
        return None, None

    candidate_perms: List[List[int]] = []
    for axis in range(rank):
        perm = [int(v) for v in range(rank) if int(v) != int(axis)] + [int(axis)]
        if perm != list(range(rank)):
            candidate_perms.append(perm)

    import itertools

    for generic_perm in itertools.permutations(range(rank)):
        perm = [int(v) for v in generic_perm]
        if perm == list(range(rank)) or perm in candidate_perms:
            continue
        candidate_perms.append(perm)

    for pre_perm in candidate_perms:
        permuted_input_shape = [int(input_shape[int(idx)]) for idx in pre_perm]
        if permuted_input_shape != value_shape:
            continue
        inverse_perm = [0] * rank
        for new_axis, old_axis in enumerate(pre_perm):
            inverse_perm[int(old_axis)] = int(new_axis)
        if index_shape is None or index_shape == value_shape:
            return pre_perm, None
        if [int(value_shape[int(idx)]) for idx in inverse_perm] == index_shape:
            return pre_perm, inverse_perm
    return None, None

def _pad_literal_expr_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[str]:
    return _torch_pad_literal_for_constant_tensor(model_ir.tensors.get(str(tensor_name), None))

def _constant_pad_pairs_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[List[int]]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    try:
        values = np.asarray(tensor.data, dtype=np.int64).reshape(-1, 2)
    except Exception:
        return None
    return [[int(v) for v in list(row)] for row in values.tolist()]

def _scalar_literal_expr_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[str]:
    return _scalar_literal_for_constant_tensor(model_ir.tensors.get(str(tensor_name), None))

def _int_scalar_literal_expr_for_codegen(
    *,
    static_int_tensor_values_fn: Callable[[str], Optional[List[int]]],
    tensor_name: str,
) -> Optional[str]:
    values = static_int_tensor_values_fn(str(tensor_name))
    if values is None or len(values) != 1:
        return None
    return repr(int(values[0]))

def _axis_expr_from_input_for_codegen(
    *,
    runtime_imports: Set[str],
    int_scalar_literal_expr_fn: Callable[[str], Optional[str]],
    tensor_expr_fn: Callable[[str], str],
    tensor_name: str,
    device_expr: str,
) -> str:
    axis_literal = int_scalar_literal_expr_fn(tensor_name)
    if axis_literal is not None:
        return axis_literal
    runtime_imports.add("_coerce_scalar_axis")
    return f"_coerce_scalar_axis({tensor_expr_fn(str(tensor_name))}, device={device_expr}.device)"

def _activation_lines_for_codegen(var_name: str, fused: str) -> List[str]:
    key = str(fused).upper()
    if key in {"", "NONE"}:
        return []
    if key == "RELU":
        return [f"{var_name} = torch.relu({var_name})"]
    if key == "RELU6":
        return [f"{var_name} = torch.clamp({var_name}, min=0.0, max=6.0)"]
    if key == "RELU_N1_TO_1":
        return [f"{var_name} = torch.clamp({var_name}, min=-1.0, max=1.0)"]
    if key == "RELU_0_TO_1":
        return [f"{var_name} = torch.clamp({var_name}, min=0.0, max=1.0)"]
    if key == "SILU":
        return [f"{var_name} = torch.mul({var_name}, torch.sigmoid({var_name}))"]
    if key == "TANH":
        return [f"{var_name} = torch.tanh({var_name})"]
    return [f"{var_name} = _apply_fused_activation({var_name}, {fused!r})"]

def _static_mirror_pad_expr_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_imports: Set[str],
    constant_pad_pairs_fn: Callable[[str], Optional[List[List[int]]]],
    tensor_expr_fn: Callable[[str], str],
    input_tensor_name: str,
    pads_tensor_name: str,
) -> Optional[str]:
    pad_pairs = constant_pad_pairs_fn(pads_tensor_name)
    input_tensor = model_ir.tensors.get(str(input_tensor_name), None)
    if pad_pairs is None or input_tensor is None:
        return None
    rank = len(list(input_tensor.shape))
    if rank <= 0:
        return None
    if len(pad_pairs) < rank:
        pad_pairs = ([[0, 0]] * (rank - len(pad_pairs))) + pad_pairs
    elif len(pad_pairs) > rank:
        pad_pairs = pad_pairs[-rank:]
    non_zero_axes = [
        idx for idx, (before, after) in enumerate(pad_pairs)
        if int(before) != 0 or int(after) != 0
    ]
    input_expr = tensor_expr_fn(str(input_tensor_name))
    if len(non_zero_axes) == 0:
        return input_expr
    if len(non_zero_axes) > 3:
        return None
    keep_axes = [idx for idx in range(rank) if idx not in non_zero_axes]
    perm = keep_axes + non_zero_axes
    expr = input_expr
    if perm != list(range(rank)):
        runtime_imports.add("_torch_permute")
        expr = f"_torch_permute({expr}, {repr(perm)})"
    torch_pad: List[int] = []
    for axis in reversed(non_zero_axes):
        before, after = pad_pairs[axis]
        torch_pad.extend([int(before), int(after)])
    expr = f"F.pad({expr}, {repr(torch_pad)}, mode='reflect')"
    if perm == list(range(rank)):
        return expr
    inverse_perm = [0] * rank
    for permuted_axis, original_axis in enumerate(perm):
        inverse_perm[int(original_axis)] = int(permuted_axis)
    runtime_imports.add("_torch_permute")
    return f"_torch_permute({expr}, {repr(inverse_perm)})"

def _is_sequential_single_input_graph_for_codegen(
    *,
    model_ir: ModelIR,
) -> bool:
    if len(model_ir.inputs) != 1 or len(model_ir.outputs) != 1:
        return False
    current_name = str(model_ir.inputs[0])
    for op in model_ir.operators:
        if len(op.outputs) != 1:
            return False
        data_input_index = 2 if str(op.op_type) in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} else 0
        if len(op.inputs) <= data_input_index:
            return False
        if str(op.inputs[data_input_index]) != current_name:
            return False
        for input_index, input_name in enumerate(op.inputs):
            if int(input_index) == int(data_input_index) or str(input_name) == "":
                continue
            input_tensor = model_ir.tensors.get(str(input_name), None)
            if input_tensor is None or not isinstance(input_tensor.data, np.ndarray):
                return False
        current_name = str(op.outputs[0])
    return current_name == str(model_ir.outputs[0])

def _is_channel_last_layout_for_codegen(logical_layout: Any) -> bool:
    return str(logical_layout).upper() in {"NWC", "NHWC", "NDHWC"}

def _tensor_dtype_name_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[str]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    return str(tensor.dtype).upper()

def _can_emit_direct_module_call_for_codegen(
    *,
    model_ir: ModelIR,
    is_channel_last_layout_fn: Callable[[Any], bool],
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    if op_type not in {"CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D"}:
        return False
    if len(op.outputs) != 1:
        return False
    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if input_tensor is None or output_tensor is None:
        return False
    if is_channel_last_layout_fn(input_tensor.logical_layout) or is_channel_last_layout_fn(output_tensor.logical_layout):
        return False
    expected_rank = 5 if op_type == "CONV_3D" else 4
    if len(input_tensor.shape) != expected_rank or len(output_tensor.shape) != expected_rank:
        return False
    if int(input_tensor.shape[1]) <= 0 or int(output_tensor.shape[1]) <= 0:
        return False
    return True

def _gather_input_pre_permute_for_codegen(
    *,
    model_ir: ModelIR,
    params_name: str,
    output_name: str,
    axis: int,
    batch_dims: int,
) -> Optional[List[int]]:
    if int(batch_dims) != 0:
        return None
    params_tensor = model_ir.tensors.get(str(params_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if params_tensor is None or output_tensor is None:
        return None
    params_shape = [int(v) for v in list(params_tensor.shape)]
    rank = len(params_shape)
    if rank not in {3, 4, 5}:
        return None
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None and len(list(output_tensor.shape_signature)) == rank
        else [int(v) for v in list(output_tensor.shape)]
    )
    if len(output_signature) != rank:
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += rank
    if resolved_axis < 0 or resolved_axis >= rank:
        return None

    def _matches_signature(shape: Sequence[int]) -> bool:
        for dim_idx, expected_dim in enumerate(output_signature):
            if dim_idx == resolved_axis or int(expected_dim) <= 0:
                continue
            if int(shape[dim_idx]) != int(expected_dim):
                return False
        return True

    if _matches_signature(params_shape):
        return None
    for perm in (_perm_cf_to_cl(rank), _perm_cl_to_cf(rank)):
        if perm is None:
            continue
        perm_values = [int(v) for v in list(perm)]
        permuted_shape = [int(params_shape[int(idx)]) for idx in perm_values]
        if _matches_signature(permuted_shape):
            return perm_values
    return None

def _infer_conv2d_layout_candidate_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    depthwise: bool,
) -> Optional[Tuple[List[int], int, int]]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
        return None
    if in_shape[0] > 0 and out_shape[0] > 0 and int(in_shape[0]) != int(out_shape[0]):
        return None
    out_channels = int(kernel_shape[0])
    if out_shape[1] > 0 and out_channels > 0 and int(out_shape[1]) != int(out_channels):
        return None
    stride_hw = [
        int((options or {}).get("strideH", 1)),
        int((options or {}).get("strideW", 1)),
    ]
    dilation_hw = [
        int((options or {}).get("dilationHFactor", 1)),
        int((options or {}).get("dilationWFactor", 1)),
    ]
    padding_mode = str((options or {}).get("padding", "SAME"))
    import itertools

    for tail_perm in itertools.permutations((1, 2, 3)):
        perm = [0, *[int(v) for v in tail_perm]]
        permuted_shape = [int(in_shape[idx]) for idx in perm]
        in_channels = int(permuted_shape[1])
        if in_channels <= 0:
            continue
        if depthwise:
            if int(kernel_shape[1]) != 1 or int(out_channels) % int(in_channels) != 0:
                continue
            groups = int(in_channels)
        else:
            weight_in_channels = int(kernel_shape[1])
            if weight_in_channels <= 0 or int(in_channels) % int(weight_in_channels) != 0:
                continue
            groups = int(in_channels) // int(weight_in_channels)
            if groups <= 0 or int(out_channels) % int(groups) != 0:
                continue
        expected_output_hw = _conv2d_output_spatial_shape_for_codegen(
            input_hw=[int(permuted_shape[2]), int(permuted_shape[3])],
            kernel_hw=[int(kernel_shape[2]), int(kernel_shape[3])],
            stride_hw=stride_hw,
            dilation_hw=dilation_hw,
            padding_mode=padding_mode,
        )
        if expected_output_hw is None:
            continue
        if (
            out_shape[2] > 0
            and out_shape[3] > 0
            and expected_output_hw != [int(out_shape[2]), int(out_shape[3])]
        ):
            continue
        return (perm, int(in_channels), int(groups))
    return None

def _conv2d_input_pre_permute_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
    depthwise: bool = False,
) -> Optional[List[int]]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
        return None
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    if is_channel_last_logical_layout(normalized_input_layout):
        return [0, 3, 1, 2]
    if is_channel_last_logical_layout(normalized_output_layout):
        return [0, 3, 1, 2]
    if depthwise:
        depthwise_channels = int(kernel_shape[0])
        if (
            depthwise_channels > 0
            and int(in_shape[1]) == depthwise_channels
            and int(out_shape[1]) == depthwise_channels
        ):
            return None
        if (
            depthwise_channels > 0
            and int(in_shape[3]) == depthwise_channels
            and int(out_shape[3]) == depthwise_channels
        ):
            return [0, 3, 1, 2]
    if (
        depthwise
        and int(kernel_shape[1]) == 1
        and int(in_shape[1]) == int(kernel_shape[0])
        and int(out_shape[1]) == int(kernel_shape[0])
    ):
        return None
    candidate_channels: List[int] = []
    for candidate in (int(in_shape[1]), int(in_shape[3])):
        if candidate > 0 and candidate not in candidate_channels:
            candidate_channels.append(candidate)
    expected_in_channels = None
    best_groups = None
    for candidate in candidate_channels:
        if int(kernel_shape[1]) <= 0 or int(candidate) % int(kernel_shape[1]) != 0:
            continue
        inferred_groups = int(candidate) // int(kernel_shape[1])
        if inferred_groups <= 0 or int(kernel_shape[0]) % int(inferred_groups) != 0:
            continue
        if best_groups is None or int(inferred_groups) < int(best_groups):
            expected_in_channels = int(candidate)
            best_groups = int(inferred_groups)
    if expected_in_channels is None:
        expected_in_channels = int(kernel_shape[1])
    if int(in_shape[1]) != expected_in_channels and int(in_shape[3]) == expected_in_channels:
        return [0, 3, 1, 2]
    if (
        kernel_shape[2] == 1
        and kernel_shape[3] > 1
        and in_shape[2] > 1
        and in_shape[3] == 1
        and out_shape[2] == 1
        and out_shape[3] > 1
    ):
        return [0, 1, 3, 2]

    inferred_layout = _infer_conv2d_layout_candidate_for_codegen(
        input_shape=in_shape,
        output_shape=out_shape,
        weight_shape=kernel_shape,
        options=options,
        depthwise=depthwise,
    )
    if inferred_layout is None:
        if is_channel_first_logical_layout(normalized_input_layout):
            return None
        return None
    perm, _, _ = inferred_layout
    if perm == [0, 1, 2, 3]:
        return None
    return perm

def _infer_effective_rank4_runtime_layout_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    consumer_index: Dict[str, List[int]],
    tensor_name: str,
) -> Optional[str]:
    current_name = str(tensor_name)
    visited: Set[str] = set()
    passthrough_ops = {
        "ADD",
        "AVERAGE_POOL_2D",
        "CAST",
        "IDENTITY",
        "LEAKY_RELU",
        "LOGISTIC",
        "MAX_POOL_2D",
        "MUL",
        "PAD",
        "PADV2",
        "RELU",
        "RELU6",
        "RELU_N1_TO_1",
        "RELU_0_TO_1",
        "TANH",
    }
    while current_name not in visited:
        visited.add(current_name)
        current_tensor = model_ir.tensors.get(current_name, None)
        if current_tensor is None or len(list(current_tensor.shape)) != 4:
            return None
        current_shape = [int(v) for v in list(current_tensor.shape)]
        current_layout = normalize_logical_layout(current_tensor.logical_layout)
        if is_channel_last_logical_layout(current_layout):
            return "NHWC"
        if is_channel_first_logical_layout(current_layout):
            return "NCHW"

        producer_idx = producer_index.get(current_name, None)
        if producer_idx is None:
            consumer_indices = consumer_index.get(current_name, [])
            if len(consumer_indices) != 1:
                return None
            consumer_op = model_ir.operators[int(consumer_indices[0])]
            if str(consumer_op.op_type) != "CONV_2D" or len(consumer_op.inputs) < 2:
                return None
            filter_tensor = model_ir.tensors.get(str(consumer_op.inputs[1]), None)
            if filter_tensor is None or len(list(filter_tensor.shape)) != 4:
                return None
            filter_shape = [int(v) for v in list(filter_tensor.shape)]
            input_channels = int(filter_shape[3])
            if current_shape[3] == input_channels and current_shape[1] != input_channels:
                return "NHWC"
            if current_shape[1] == input_channels and current_shape[3] != input_channels:
                return "NCHW"
            return None

        producer_op = model_ir.operators[int(producer_idx)]
        producer_type = str(producer_op.op_type)
        if producer_type == "CONV_2D" and len(producer_op.inputs) >= 2:
            filter_tensor = model_ir.tensors.get(str(producer_op.inputs[1]), None)
            if filter_tensor is None or len(list(filter_tensor.shape)) != 4:
                return None
            filter_shape = [int(v) for v in list(filter_tensor.shape)]
            out_channels = int(filter_shape[0])
            if current_shape[3] == out_channels and current_shape[1] != out_channels:
                return "NHWC"
            if current_shape[1] == out_channels and current_shape[3] != out_channels:
                return "NCHW"
            return None
        if producer_type not in passthrough_ops or len(producer_op.inputs) <= 0:
            return None

        exact_shape_name = None
        permuted_shape_name = None
        fallback_name = None
        perm_cl_to_cf = _perm_cl_to_cf(4)
        perm_cf_to_cl = _perm_cf_to_cl(4)
        for candidate_input in list(producer_op.inputs):
            candidate_name = str(candidate_input)
            candidate_tensor = model_ir.tensors.get(candidate_name, None)
            if candidate_tensor is None or candidate_tensor.data is not None:
                continue
            if len(list(candidate_tensor.shape)) == 4:
                candidate_shape = [int(v) for v in list(candidate_tensor.shape)]
                if candidate_shape == current_shape:
                    exact_shape_name = candidate_name
                    break
                if (
                    perm_cl_to_cf is not None
                    and _permute_shape(candidate_shape, perm_cl_to_cf) == current_shape
                ) or (
                    perm_cf_to_cl is not None
                    and _permute_shape(candidate_shape, perm_cf_to_cl) == current_shape
                ):
                    if permuted_shape_name is None:
                        permuted_shape_name = candidate_name
                    continue
                if fallback_name is None:
                    fallback_name = candidate_name
        next_name = exact_shape_name or permuted_shape_name or fallback_name
        if next_name is None:
            return None
        current_name = next_name
    return None

def _infer_conv2d_ctor_params_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
    depthwise: bool,
) -> Tuple[int, int]:
    if input_shape is None or weight_shape is None:
        return (1, 1)
    in_shape = [int(v) for v in list(input_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 4 or len(kernel_shape) != 4:
        return (1, 1)
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    preferred_input_channels: Optional[int] = None

    def _choose_unknown_input_channel_candidate() -> Optional[int]:
        candidate_channels: List[int] = []
        for candidate in (int(in_shape[1]), int(in_shape[3])):
            if candidate > 0 and candidate not in candidate_channels:
                candidate_channels.append(candidate)
        if len(candidate_channels) == 0:
            return None
        out_channels = max(1, int(kernel_shape[0]))
        if depthwise:
            valid_candidates = [
                int(candidate)
                for candidate in candidate_channels
                if int(candidate) > 0 and int(out_channels) % int(candidate) == 0
            ]
            if len(valid_candidates) > 0:
                return int(max(valid_candidates))
            return None
        weight_in_channels = max(1, int(kernel_shape[1]))
        best_choice: Optional[Tuple[int, int]] = None
        for candidate in candidate_channels:
            if int(candidate) % int(weight_in_channels) != 0:
                continue
            groups = int(candidate) // int(weight_in_channels)
            if groups <= 0 or int(out_channels) % int(groups) != 0:
                continue
            choice = (int(candidate), int(groups))
            if best_choice is None or int(choice[1]) < int(best_choice[1]):
                best_choice = choice
        if best_choice is not None:
            return int(best_choice[0])
        return None

    if is_channel_first_logical_layout(normalized_input_layout):
        preferred_input_channels = int(in_shape[1])
    elif is_channel_last_logical_layout(normalized_input_layout):
        preferred_input_channels = int(in_shape[3])
    else:
        preferred_input_channels = _choose_unknown_input_channel_candidate()
        if preferred_input_channels is None and output_shape is not None:
            out_shape = [int(v) for v in list(output_shape)]
            if len(out_shape) == 4:
                if is_channel_first_logical_layout(normalized_output_layout):
                    preferred_input_channels = int(in_shape[1])
                elif is_channel_last_logical_layout(normalized_output_layout):
                    preferred_input_channels = int(in_shape[3])
    if preferred_input_channels is not None and int(preferred_input_channels) > 0:
        if depthwise:
            out_channels = max(1, int(kernel_shape[0]))
            if int(out_channels) % int(preferred_input_channels) == 0:
                return (int(preferred_input_channels), int(preferred_input_channels))
        weight_in_channels = max(1, int(kernel_shape[1]))
        if int(preferred_input_channels) % int(weight_in_channels) == 0:
            preferred_groups = int(preferred_input_channels) // int(weight_in_channels)
            out_channels = max(1, int(kernel_shape[0]))
            if preferred_groups > 0 and int(out_channels) % int(preferred_groups) == 0:
                return (int(preferred_input_channels), int(preferred_groups))
    inferred_layout = _infer_conv2d_layout_candidate_for_codegen(
        input_shape=in_shape,
        output_shape=output_shape,
        weight_shape=kernel_shape,
        options=options,
        depthwise=depthwise,
    )
    if inferred_layout is not None:
        _, inferred_in_channels, inferred_groups = inferred_layout
        return (max(1, int(inferred_in_channels)), max(1, int(inferred_groups)))
    candidate_channels: List[int] = []
    for candidate in (int(in_shape[1]), int(in_shape[3])):
        if candidate > 0 and candidate not in candidate_channels:
            candidate_channels.append(candidate)
    out_channels = max(1, int(kernel_shape[0]))
    if depthwise:
        valid_candidates = [
            int(candidate)
            for candidate in candidate_channels
            if int(candidate) > 0 and int(out_channels) % int(candidate) == 0
        ]
        if len(valid_candidates) == 0:
            inferred_in_channels = int(candidate_channels[-1]) if len(candidate_channels) > 0 else 1
        else:
            inferred_in_channels = int(max(valid_candidates))
        return (max(1, inferred_in_channels), max(1, inferred_in_channels))

    weight_in_channels = max(1, int(kernel_shape[1]))
    best_choice: Optional[Tuple[int, int]] = None
    for candidate in candidate_channels:
        if int(candidate) % int(weight_in_channels) != 0:
            continue
        groups = int(candidate) // int(weight_in_channels)
        if groups <= 0 or int(out_channels) % int(groups) != 0:
            continue
        choice = (int(candidate), int(groups))
        if best_choice is None or int(choice[1]) < int(best_choice[1]):
            best_choice = choice
    if best_choice is not None:
        return best_choice
    fallback_in_channels = int(candidate_channels[-1]) if len(candidate_channels) > 0 else int(weight_in_channels)
    return (max(1, fallback_in_channels), 1)

def _infer_conv3d_ctor_params_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
) -> Tuple[int, int, int, List[int]]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return (1, 1, 1, [1, 1, 1])
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 5 or len(out_shape) != 5 or len(kernel_shape) != 5:
        return (1, 1, max(1, int(kernel_shape[0]) if len(kernel_shape) > 0 else 1), [1, 1, 1])
    if in_shape[0] > 0 and out_shape[0] > 0 and int(in_shape[0]) != int(out_shape[0]):
        return (max(1, int(in_shape[1])), 1, max(1, int(out_shape[1])), [int(v) for v in list(kernel_shape[2:5])])
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    input_channels = max(
        1,
        int(in_shape[-1]) if is_channel_last_logical_layout(normalized_input_layout) else int(in_shape[1]),
    )
    expected_out_channels = max(
        1,
        int(out_shape[-1]) if is_channel_last_logical_layout(normalized_output_layout) else int(out_shape[1]),
    )
    input_dhw = (
        [int(in_shape[1]), int(in_shape[2]), int(in_shape[3])]
        if is_channel_last_logical_layout(normalized_input_layout)
        else [int(in_shape[2]), int(in_shape[3]), int(in_shape[4])]
    )
    output_dhw = (
        [int(out_shape[1]), int(out_shape[2]), int(out_shape[3])]
        if is_channel_last_logical_layout(normalized_output_layout)
        else [int(out_shape[2]), int(out_shape[3]), int(out_shape[4])]
    )
    stride_dhw = [
        int((options or {}).get("strideD", 1)),
        int((options or {}).get("strideH", 1)),
        int((options or {}).get("strideW", 1)),
    ]
    dilation_dhw = [
        int((options or {}).get("dilationDFactor", 1)),
        int((options or {}).get("dilationHFactor", 1)),
        int((options or {}).get("dilationWFactor", 1)),
    ]
    padding_mode = str((options or {}).get("padding", "SAME"))

    import itertools

    best_choice: Optional[Tuple[int, int, int, List[int]]] = None
    for out_axis in range(5):
        out_channels = int(kernel_shape[out_axis])
        if out_channels <= 0 or out_channels != expected_out_channels:
            continue
        for in_axis in range(5):
            if in_axis == out_axis:
                continue
            weight_in_channels = int(kernel_shape[in_axis])
            if weight_in_channels <= 0 or int(input_channels) % int(weight_in_channels) != 0:
                continue
            groups = int(input_channels) // int(weight_in_channels)
            if groups <= 0 or int(out_channels) % int(groups) != 0:
                continue
            kernel_axes = [idx for idx in range(5) if idx not in {out_axis, in_axis}]
            if len(kernel_axes) != 3:
                continue
            for kernel_order in itertools.permutations(kernel_axes):
                kernel_dhw = [int(kernel_shape[idx]) for idx in kernel_order]
                expected_output_dhw = _conv3d_output_spatial_shape_for_codegen(
                    input_dhw=input_dhw,
                    kernel_dhw=kernel_dhw,
                    stride_dhw=stride_dhw,
                    dilation_dhw=dilation_dhw,
                    padding_mode=padding_mode,
                )
                if expected_output_dhw is None:
                    continue
                if expected_output_dhw != output_dhw:
                    continue
                choice = (int(input_channels), int(groups), int(out_channels), [int(v) for v in kernel_dhw])
                if best_choice is None or int(choice[1]) < int(best_choice[1]):
                    best_choice = choice
    if best_choice is not None:
        return best_choice
    fallback_kernel = [int(v) for v in list(kernel_shape[2:5])]
    return (max(1, int(input_channels)), 1, max(1, int(expected_out_channels)), fallback_kernel)

def _infer_conv3d_transpose_ctor_params_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
) -> Tuple[int, int, List[int], int]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return (1, 1, [1, 1, 1], 1)
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 5 or len(out_shape) != 5 or len(kernel_shape) != 5:
        return (1, max(1, int(out_shape[1]) if len(out_shape) > 1 else 1), [1, 1, 1], 1)
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    input_channels = max(
        1,
        int(in_shape[-1]) if is_channel_last_logical_layout(normalized_input_layout) else int(in_shape[1]),
    )
    expected_out_channels = max(
        1,
        int(out_shape[-1]) if is_channel_last_logical_layout(normalized_output_layout) else int(out_shape[1]),
    )
    input_dhw = (
        [int(in_shape[1]), int(in_shape[2]), int(in_shape[3])]
        if is_channel_last_logical_layout(normalized_input_layout)
        else [int(in_shape[2]), int(in_shape[3]), int(in_shape[4])]
    )
    output_dhw = (
        [int(out_shape[1]), int(out_shape[2]), int(out_shape[3])]
        if is_channel_last_logical_layout(normalized_output_layout)
        else [int(out_shape[2]), int(out_shape[3]), int(out_shape[4])]
    )
    stride_dhw = [
        int((options or {}).get("strideD", 1)),
        int((options or {}).get("strideH", 1)),
        int((options or {}).get("strideW", 1)),
    ]
    dilation_dhw = [
        int((options or {}).get("dilationDFactor", 1)),
        int((options or {}).get("dilationHFactor", 1)),
        int((options or {}).get("dilationWFactor", 1)),
    ]
    padding_mode = str((options or {}).get("padding", "SAME"))

    import itertools

    best_choice: Optional[Tuple[int, int, List[int], int]] = None
    for in_axis in range(5):
        weight_in_channels = int(kernel_shape[in_axis])
        if weight_in_channels <= 0 or weight_in_channels != input_channels:
            continue
        for out_axis in range(5):
            if out_axis == in_axis:
                continue
            weight_out_per_group = int(kernel_shape[out_axis])
            if weight_out_per_group <= 0 or int(expected_out_channels) % int(weight_out_per_group) != 0:
                continue
            groups = int(expected_out_channels) // int(weight_out_per_group)
            if groups <= 0 or int(input_channels) % int(groups) != 0:
                continue
            kernel_axes = [idx for idx in range(5) if idx not in {in_axis, out_axis}]
            if len(kernel_axes) != 3:
                continue
            for kernel_order in itertools.permutations(kernel_axes):
                kernel_dhw = [int(kernel_shape[idx]) for idx in kernel_order]
                expected_output_dhw = _conv3d_transpose_output_spatial_shape_for_codegen(
                    input_dhw=input_dhw,
                    kernel_dhw=kernel_dhw,
                    stride_dhw=stride_dhw,
                    dilation_dhw=dilation_dhw,
                    padding_mode=padding_mode,
                )
                if expected_output_dhw is None:
                    continue
                if expected_output_dhw != output_dhw:
                    continue
                choice = (
                    int(input_channels),
                    int(expected_out_channels),
                    [int(v) for v in kernel_dhw],
                    int(groups),
                )
                if best_choice is None or int(choice[3]) < int(best_choice[3]):
                    best_choice = choice
    if best_choice is not None:
        return best_choice
    fallback_kernel = [int(v) for v in list(kernel_shape[2:5])]
    return (max(1, int(input_channels)), max(1, int(expected_out_channels)), fallback_kernel, 1)

def _binary_trailing_axis_constant_buffer_alias_shape_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    inlined_constant_tensor_names: Set[str],
    tensor_name: str,
    other_tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
    if tensor is None or other_tensor is None:
        return None
    if not isinstance(tensor.data, np.ndarray):
        return None
    if str(tensor_name) in model_ir.inputs or str(tensor_name) in producer_index:
        return None
    if str(tensor_name) in inlined_constant_tensor_names:
        return None
    tensor_shape = [int(v) for v in list(tensor.shape)]
    other_shape = [int(v) for v in list(other_tensor.shape)]
    if len(tensor_shape) != 1 or len(other_shape) < 2:
        return None
    constant_width = int(tensor_shape[0])
    if constant_width <= 0:
        return None
    target_axis: Optional[int] = None
    other_layout = normalize_logical_layout(other_tensor.logical_layout)
    if (
        is_channel_first_logical_layout(other_layout)
        and len(other_shape) >= 2
        and int(other_shape[1]) == constant_width
    ):
        target_axis = 1
    elif (
        is_channel_last_logical_layout(other_layout)
        and int(other_shape[-1]) == constant_width
    ):
        target_axis = len(other_shape) - 1
    else:
        matching_axes = [
            int(axis)
            for axis, dim in enumerate(other_shape)
            if int(axis) != 0 and int(dim) == constant_width
        ]
        if len(matching_axes) == 1:
            target_axis = int(matching_axes[0])
    if target_axis is None or int(target_axis) != len(other_shape) - 1:
        return None
    return [1 for _ in range(len(other_shape) - 1)] + [int(constant_width)]

def _channel_first_rank4_constant_buffer_alias_shape_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    inlined_constant_tensor_names: Set[str],
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    if str(tensor_name) in model_ir.inputs or str(tensor_name) in producer_index:
        return None
    if str(tensor_name) in inlined_constant_tensor_names:
        return None
    if bool(tensor.is_variable):
        return None
    tensor_shape = [int(v) for v in list(tensor.shape)]
    if len(tensor_shape) != 4:
        return None
    if tensor_shape[0] != 1 or tensor_shape[1] != 1 or tensor_shape[2] != 1 or tensor_shape[3] <= 0:
        return None
    return [1, int(tensor_shape[3]), 1, 1]

def _constant_permute_for_broadcast_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    other_tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
    if tensor is None or other_tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    tensor_shape = [int(v) for v in list(tensor.shape)]
    other_shape = [int(v) for v in list(other_tensor.shape)]
    if len(tensor_shape) != len(other_shape) or len(tensor_shape) <= 1:
        return None
    if _shape_lists_equal_relaxed(tensor_shape, other_shape):
        return None
    tensor_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in tensor_shape]
    other_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in other_shape]
    try:
        np.broadcast_shapes(tuple(tensor_broadcast_shape), tuple(other_broadcast_shape))
        non_singleton_axes = [idx for idx, dim in enumerate(tensor_shape) if int(dim) > 1]
        if (
            len(non_singleton_axes) == 1
            and int(non_singleton_axes[0]) == len(tensor_shape) - 1
            and int(tensor_shape[-1]) == int(other_shape[-1])
        ):
            return None
    except Exception:
        pass

    def _try_exact_match(perm: Sequence[int]) -> Optional[List[int]]:
        permuted_shape = [int(tensor_shape[int(idx)]) for idx in list(perm)]
        if [int(v) for v in list(permuted_shape)] == [int(v) for v in list(other_shape)]:
            return [int(v) for v in list(perm)]
        return None

    preferred_perm: Optional[Tuple[int, ...]] = None
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    if tensor_layout == "NCHW" and len(tensor_shape) == 4:
        preferred_perm = (0, 2, 3, 1)
    elif tensor_layout == "NCDHW" and len(tensor_shape) == 5:
        preferred_perm = (0, 2, 3, 4, 1)
    elif tensor_layout == "NCW" and len(tensor_shape) == 3:
        preferred_perm = (0, 2, 1)
    if preferred_perm is not None:
        exact_perm = _try_exact_match(preferred_perm)
        if exact_perm is not None:
            return exact_perm
        preferred_shape = [int(tensor_shape[int(idx)]) for idx in preferred_perm]
        preferred_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in preferred_shape]
        try:
            np.broadcast_shapes(tuple(preferred_broadcast_shape), tuple(other_broadcast_shape))
            return [int(v) for v in list(preferred_perm)]
        except Exception:
            pass

    import itertools

    for generic_perm in itertools.permutations(range(len(tensor_shape))):
        if list(generic_perm) == list(range(len(tensor_shape))):
            continue
        exact_perm = _try_exact_match(generic_perm)
        if exact_perm is not None:
            return exact_perm

    best_broadcast_perm: Optional[Tuple[int, List[int]]] = None
    for generic_perm in itertools.permutations(range(len(tensor_shape))):
        if list(generic_perm) == list(range(len(tensor_shape))):
            continue
        permuted_shape = [int(tensor_shape[int(idx)]) for idx in generic_perm]
        permuted_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in permuted_shape]
        try:
            np.broadcast_shapes(tuple(permuted_broadcast_shape), tuple(other_broadcast_shape))
            score = sum(
                1
                for permuted_dim, other_dim in zip(permuted_shape, other_shape)
                if int(permuted_dim) == int(other_dim)
            )
            candidate = (int(score), [int(v) for v in list(generic_perm)])
            if best_broadcast_perm is None or candidate[0] > best_broadcast_perm[0]:
                best_broadcast_perm = candidate
        except Exception:
            continue
    return None if best_broadcast_perm is None else list(best_broadcast_perm[1])

def _tensor_expr_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    tensor_expr_aliases: Dict[str, str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    buffer_attr_names: Dict[str, str],
    runtime_imports: Set[str],
    tensor_var_names: Dict[str, str],
    tensor_name: str,
) -> str:
    if str(tensor_name) in tensor_expr_aliases:
        return str(tensor_expr_aliases[str(tensor_name)])
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if (
        str(tensor_name) in channel_first_tensor_expr_aliases
        and tensor is not None
        and is_channel_first_logical_layout(normalize_logical_layout(tensor.logical_layout))
    ):
        return str(channel_first_tensor_expr_aliases[str(tensor_name)])
    if (
        tensor is not None
        and str(tensor_name) not in model_ir.inputs
        and str(tensor_name) not in producer_index
    ):
        if str(tensor_name) in buffer_attr_names:
            return f"self.{buffer_attr_names[str(tensor_name)]}"
        literal = _python_literal_for_constant_tensor(tensor)
        if literal is not None:
            runtime_imports.add("_module_device")
            return (
                f"torch.as_tensor({literal}, dtype={_torch_dtype_literal(str(tensor.dtype).upper())}, "
                "device=_module_device(self))"
            )
    if str(tensor_name) in tensor_var_names:
        return str(tensor_var_names[str(tensor_name)])
    if str(tensor_name) in buffer_attr_names:
        return f"self.{buffer_attr_names[str(tensor_name)]}"
    literal = _python_literal_for_constant_tensor(tensor) if tensor is not None else None
    if tensor is not None and literal is not None:
        runtime_imports.add("_module_device")
        return (
            f"torch.as_tensor({literal}, dtype={_torch_dtype_literal(str(tensor.dtype).upper())}, "
            "device=_module_device(self))"
        )
    raise ModelIRPyTorchExportError(
        "Native PyTorch-like model.py codegen could not resolve a tensor expression. "
        f"tensor={tensor_name}"
    )

def _derived_local_var_name_for_codegen(
    *,
    synthetic_local_var_names: Dict[str, str],
    used_local_var_names: Set[str],
    base_name: str,
    prefix: str = "t",
) -> str:
    cached_name = synthetic_local_var_names.get(str(base_name), None)
    if cached_name is not None:
        return str(cached_name)
    shortened_name = _shorten_generated_python_identifier(
        str(base_name),
        prefix=prefix,
    )
    unique_name = _make_unique_identifier(shortened_name, used_local_var_names)
    synthetic_local_var_names[str(base_name)] = str(unique_name)
    return str(unique_name)

def _channel_first_constant_expr_for_buffer_attr_for_codegen(
    *,
    buffer_attr_name_to_tensor_name: Dict[str, str],
    channel_first_constant_buffer_alias_exprs: Dict[str, str],
    channel_first_rank4_constant_buffer_alias_shape_fn: Callable[[str], Optional[List[int]]],
    buffer_expr: str,
    target_shape: Sequence[int],
) -> Optional[str]:
    if not str(buffer_expr).startswith("self."):
        return None
    attr_name = str(buffer_expr)[5:]
    tensor_name = buffer_attr_name_to_tensor_name.get(str(attr_name), None)
    if tensor_name is None:
        return None
    alias_expr = channel_first_constant_buffer_alias_exprs.get(str(tensor_name), None)
    alias_shape = channel_first_rank4_constant_buffer_alias_shape_fn(str(tensor_name))
    if alias_expr is None or alias_shape is None:
        return None
    if [int(v) for v in alias_shape] != [int(v) for v in list(target_shape)]:
        return None
    return str(alias_expr)

def _permuted_constant_expr_for_tensor_name_for_codegen(
    *,
    permuted_constant_buffer_alias_exprs: Dict[Tuple[str, Tuple[int, ...]], str],
    tensor_name: str,
    perm: Sequence[int],
) -> Optional[str]:
    return permuted_constant_buffer_alias_exprs.get(
        (str(tensor_name), tuple(int(v) for v in list(perm))),
        None,
    )

def _transposed_constant_expr_for_tensor_name_for_codegen(
    *,
    transposed_constant_buffer_alias_exprs: Dict[str, str],
    tensor_name: str,
) -> Optional[str]:
    return transposed_constant_buffer_alias_exprs.get(str(tensor_name), None)

def _tensor_expr_for_channel_first_bridge_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
    perm: Optional[Sequence[int]],
) -> Optional[str]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    rank = len(list(tensor.shape))
    expected_perm = _perm_cl_to_cf(rank)
    if expected_perm is None or list(perm or []) != list(expected_perm):
        return None
    return channel_first_tensor_expr_aliases.get(str(tensor_name), None)

def _channel_first_reduction_plan_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    op: OperatorIR,
    input_name: str,
) -> Optional[Tuple[str, List[int]]]:
    op_type = str(op.op_type)
    if op_type not in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
        return None
    if len(op.inputs) < 2 or str(op.inputs[0]) != str(input_name):
        return None
    input_tensor = model_ir.tensors.get(str(input_name), None)
    if input_tensor is None:
        return None
    input_rank = len(list(input_tensor.shape))
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    if not is_channel_last_logical_layout(input_layout):
        return None
    if (
        input_rank == 4
        and len(op.inputs) >= 2
        and _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None)) == [1, 2]
    ):
        input_shape = [int(v) for v in list(input_tensor.shape)]
        non_batch_dims = [dim for dim in input_shape[1:] if dim > 0]
        if len(set(non_batch_dims)) != len(non_batch_dims):
            return None
    input_expr = channel_first_tensor_expr_aliases.get(str(input_name), None)
    if input_expr is None:
        return None
    axis_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
    if axis_values is None:
        return None
    cl_spatial_axes = {
        3: [1],
        4: [1, 2],
        5: [1, 2, 3],
    }.get(input_rank, None)
    cf_spatial_axes = {
        3: [2],
        4: [2, 3],
        5: [2, 3, 4],
    }.get(input_rank, None)
    if cl_spatial_axes is None or cf_spatial_axes is None:
        return None
    if [int(v) for v in list(axis_values)] != [int(v) for v in list(cl_spatial_axes)]:
        return None
    return str(input_expr), [int(v) for v in list(cf_spatial_axes)]

def _normalized_constant_reduction_axes_for_codegen(
    *,
    axis_values: Optional[Sequence[int]],
    rank: int,
) -> Optional[List[int]]:
    if axis_values is None:
        return None
    normalized_axes: List[int] = []
    for axis in list(axis_values):
        normalized_axis = int(axis)
        if normalized_axis < 0:
            normalized_axis += int(rank)
        if normalized_axis < 0 or normalized_axis >= int(rank):
            return None
        if normalized_axis not in normalized_axes:
            normalized_axes.append(normalized_axis)
    normalized_axes.sort()
    return normalized_axes

def _direct_mean_reduction_expr_for_codegen(
    *,
    normalized_constant_reduction_axes_fn: Callable[[Optional[Sequence[int]], int], Optional[List[int]]],
    input_expr: str,
    axes: Optional[Sequence[int]],
    input_rank: int,
    keepdims: bool,
) -> Optional[str]:
    normalized_axes = normalized_constant_reduction_axes_fn(axes, input_rank)
    if normalized_axes is None:
        return None
    if len(normalized_axes) == 0:
        return input_expr
    dim_literal: Any = normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
    return f"torch.mean({input_expr}, dim={repr(dim_literal)}, keepdim={keepdims})"

def _channel_first_passthrough_input_expr_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_expr_fn: Callable[[str], str],
    tensor_name: str,
) -> Optional[str]:
    def _has_unambiguous_channel_first_shape(shape: Sequence[int]) -> bool:
        return (
            len(list(shape)) in {3, 4, 5}
            and len(list(shape)) >= 2
            and int(shape[1]) != int(shape[-1])
        )

    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    alias_expr = channel_first_tensor_expr_aliases.get(str(tensor_name), None)
    if alias_expr is not None:
        return str(alias_expr)
    if is_channel_first_logical_layout(tensor_layout):
        tensor_expr = tensor_expr_fn(str(tensor_name))
        if (
            str(tensor_expr).endswith("_cf")
            or (
                ".permute(" in str(tensor_expr)
                and ".contiguous()" in str(tensor_expr)
            )
            or _has_unambiguous_channel_first_shape(tensor.shape)
        ):
            return str(tensor_expr)
    producer_idx = producer_index.get(str(tensor_name), None)
    if producer_idx is None:
        return None
    producer_op = model_ir.operators[int(producer_idx)]
    if str(producer_op.op_type) != "TRANSPOSE" or len(producer_op.inputs) < 1:
        return None
    transpose_perm = _read_transpose_perm(model_ir, producer_op)
    expected_cf_to_cl_perm = _perm_cf_to_cl(len(list(tensor.shape)))
    if (
        expected_cf_to_cl_perm is None
        or list(transpose_perm or []) != list(expected_cf_to_cl_perm)
    ):
        return None
    return _channel_first_passthrough_input_expr_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        tensor_expr_fn=tensor_expr_fn,
        tensor_name=str(producer_op.inputs[0]),
    )

def _can_resolve_channel_first_expr_statically_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    channel_first_tensor_expr_aliases: Dict[str, str],
    direct_codegen_unary_expressions: Collection[str],
    tensor_name: str,
    seen_names: Optional[Set[str]] = None,
) -> bool:
    current_name = str(tensor_name)
    if seen_names is None:
        seen_names = set()
    if current_name in seen_names:
        return True
    next_seen = set(seen_names)
    next_seen.add(current_name)
    tensor = model_ir.tensors.get(current_name, None)
    if tensor is None:
        return False
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    if is_channel_first_logical_layout(tensor_layout):
        return True
    if current_name in channel_first_tensor_expr_aliases:
        return True
    producer_idx = producer_index.get(current_name, None)
    if producer_idx is None:
        return False
    producer_op = model_ir.operators[int(producer_idx)]
    producer_type = str(producer_op.op_type)
    if producer_type == "TRANSPOSE" and len(producer_op.inputs) >= 1:
        transpose_perm = _read_transpose_perm(model_ir, producer_op)
        expected_cf_to_cl_perm = _perm_cf_to_cl(len(list(tensor.shape)))
        if (
            expected_cf_to_cl_perm is not None
            and list(transpose_perm or []) == list(expected_cf_to_cl_perm)
        ):
            return _can_resolve_channel_first_expr_statically_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
                direct_codegen_unary_expressions=direct_codegen_unary_expressions,
                tensor_name=str(producer_op.inputs[0]),
                seen_names=next_seen,
            )
    if producer_type in {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "TRANSPOSE_CONV",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
    }:
        return True
    if producer_type in direct_codegen_unary_expressions and len(producer_op.inputs) == 1:
        return _can_resolve_channel_first_expr_statically_for_codegen(
            model_ir=model_ir,
            producer_index=producer_index,
            channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
            direct_codegen_unary_expressions=direct_codegen_unary_expressions,
            tensor_name=str(producer_op.inputs[0]),
            seen_names=next_seen,
        )
    return False

def _can_emit_channel_first_shape_preserving_unary_op_for_codegen(
    *,
    model_ir: ModelIR,
    direct_codegen_unary_expressions: Collection[str],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    can_resolve_channel_first_expr_statically_fn: Callable[[str], bool],
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    if op_type not in direct_codegen_unary_expressions:
        return False
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        return False
    input_name = str(op.inputs[0])
    output_name = str(op.outputs[0])
    input_tensor = model_ir.tensors.get(input_name, None)
    output_tensor = model_ir.tensors.get(output_name, None)
    if input_tensor is None or output_tensor is None:
        return False
    output_rank = len(list(output_tensor.shape))
    if output_rank not in {3, 4, 5}:
        return False
    input_shape = tensor_shape_list_fn(input_name)
    output_shape = tensor_shape_list_fn(output_name)
    if (
        input_shape is None
        or output_shape is None
        or not _shape_lists_equal_relaxed(input_shape, output_shape)
    ):
        return False
    return can_resolve_channel_first_expr_statically_fn(input_name)

def _match_if_axis0_tensor_mux_slice_for_codegen(
    *,
    model_ir: ModelIR,
    producer_by_output_name: Dict[str, OperatorIR],
    op: OperatorIR,
) -> Optional[Dict[str, str]]:
    if str(op.op_type) != "SLICE" or len(op.inputs) < 3:
        return None

    def _unwrap_axis0_concat_prefix(tensor_name: str) -> Optional[str]:
        producer = producer_by_output_name.get(str(tensor_name), None)
        if producer is None:
            return str(tensor_name)
        if str(producer.op_type) != "CONCATENATION" or int(producer.options.get("axis", 0)) != 0:
            return str(tensor_name)
        concat_inputs = [str(v) for v in list(producer.inputs)]
        if len(concat_inputs) != 2:
            return None
        tail_values = _constant_int_list(model_ir.tensors.get(concat_inputs[1], None))
        if tail_values is None:
            return None
        return concat_inputs[0]

    merged_name = str(op.inputs[0])
    begin_name = str(op.inputs[1])
    size_name = str(op.inputs[2])

    merged_producer = producer_by_output_name.get(merged_name, None)
    if merged_producer is None:
        return None
    if str(merged_producer.op_type) != "CONCATENATION" or int(merged_producer.options.get("axis", 0)) != 0:
        return None
    merged_inputs = [str(v) for v in list(merged_producer.inputs)]
    if len(merged_inputs) != 2:
        return None

    begin_axis0_name = _unwrap_axis0_concat_prefix(begin_name)
    size_axis0_name = _unwrap_axis0_concat_prefix(size_name)
    if begin_axis0_name is None or size_axis0_name is None:
        return None

    begin_axis0_producer = producer_by_output_name.get(begin_axis0_name, None)
    size_axis0_producer = producer_by_output_name.get(size_axis0_name, None)
    if begin_axis0_producer is None or size_axis0_producer is None:
        return None
    if str(begin_axis0_producer.op_type) != "MUL" or str(size_axis0_producer.op_type) != "ADD":
        return None

    begin_inputs = [str(v) for v in list(begin_axis0_producer.inputs)]
    size_axis0_inputs = [str(v) for v in list(size_axis0_producer.inputs)]
    if len(begin_inputs) != 2 or len(size_axis0_inputs) != 2:
        return None

    not_cond_i32_name = begin_inputs[0]
    then_first_dim_name = begin_inputs[1]
    size_then_name = size_axis0_inputs[0]
    size_else_name = size_axis0_inputs[1]

    size_then_producer = producer_by_output_name.get(size_then_name, None)
    size_else_producer = producer_by_output_name.get(size_else_name, None)
    if size_then_producer is None or size_else_producer is None:
        return None
    if str(size_then_producer.op_type) != "MUL" or str(size_else_producer.op_type) != "MUL":
        return None

    size_then_inputs = [str(v) for v in list(size_then_producer.inputs)]
    size_else_inputs = [str(v) for v in list(size_else_producer.inputs)]
    if len(size_then_inputs) != 2 or len(size_else_inputs) != 2:
        return None

    cond_i32_name = size_then_inputs[0]
    size_then_first_dim_name = size_then_inputs[1]
    size_else_not_cond_name = size_else_inputs[0]
    else_first_dim_name = size_else_inputs[1]
    if (
        then_first_dim_name != size_then_first_dim_name
        or not_cond_i32_name != size_else_not_cond_name
    ):
        return None

    not_cond_i32_producer = producer_by_output_name.get(not_cond_i32_name, None)
    cond_i32_producer = producer_by_output_name.get(cond_i32_name, None)
    if not_cond_i32_producer is None or cond_i32_producer is None:
        return None
    if str(not_cond_i32_producer.op_type) != "SUB" or str(cond_i32_producer.op_type) != "CAST":
        return None

    not_cond_inputs = [str(v) for v in list(not_cond_i32_producer.inputs)]
    cond_inputs = [str(v) for v in list(cond_i32_producer.inputs)]
    if len(not_cond_inputs) != 2 or len(cond_inputs) != 1:
        return None
    if not_cond_inputs[1] != cond_i32_name:
        return None
    cond_name = cond_inputs[0]

    then_first_dim_values = _constant_int_list(model_ir.tensors.get(then_first_dim_name, None))
    else_first_dim_values = _constant_int_list(model_ir.tensors.get(else_first_dim_name, None))
    if (
        then_first_dim_values is None
        or else_first_dim_values is None
        or len(then_first_dim_values) != 1
        or len(else_first_dim_values) != 1
    ):
        return None

    return {
        "cond_name": cond_name,
        "then_name": merged_inputs[0],
        "else_name": merged_inputs[1],
    }

def _binary_runtime_shape_passthrough_operand_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_shape_uncertain_tensors: Set[str],
    lhs_name: str,
    rhs_name: str,
) -> Optional[str]:
    lhs_tensor = model_ir.tensors.get(str(lhs_name), None)
    rhs_tensor = model_ir.tensors.get(str(rhs_name), None)
    if lhs_tensor is None or rhs_tensor is None:
        return None
    lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
    rhs_shape = [int(v) for v in list(rhs_tensor.shape)]
    lhs_signature = (
        [int(v) for v in list(lhs_tensor.shape_signature)]
        if lhs_tensor.shape_signature is not None
        else list(lhs_shape)
    )
    rhs_signature = (
        [int(v) for v in list(rhs_tensor.shape_signature)]
        if rhs_tensor.shape_signature is not None
        else list(rhs_shape)
    )
    if (
        str(lhs_name) in runtime_shape_uncertain_tensors
        and _is_all_ones_shape(rhs_shape)
        and _is_all_ones_shape(rhs_signature)
    ):
        return "lhs"
    if (
        str(rhs_name) in runtime_shape_uncertain_tensors
        and _is_all_ones_shape(lhs_shape)
        and _is_all_ones_shape(lhs_signature)
    ):
        return "rhs"
    return None

def _binary_requires_runtime_alignment_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_shape_uncertain_tensors: Set[str],
    lhs_name: str,
    rhs_name: str,
    output_name: str,
) -> bool:
    lhs_tensor = model_ir.tensors.get(str(lhs_name), None)
    rhs_tensor = model_ir.tensors.get(str(rhs_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if lhs_tensor is None or rhs_tensor is None:
        return False
    if (
        str(lhs_name) in runtime_shape_uncertain_tensors
        or str(rhs_name) in runtime_shape_uncertain_tensors
        or str(output_name) in runtime_shape_uncertain_tensors
    ):
        return True
    lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
    rhs_shape = [int(v) for v in list(rhs_tensor.shape)]
    lhs_signature = (
        [int(v) for v in list(lhs_tensor.shape_signature)]
        if lhs_tensor.shape_signature is not None
        else list(lhs_shape)
    )
    rhs_signature = (
        [int(v) for v in list(rhs_tensor.shape_signature)]
        if rhs_tensor.shape_signature is not None
        else list(rhs_shape)
    )
    if len(lhs_shape) != len(rhs_shape):
        return False
    if lhs_shape == rhs_shape and lhs_signature != rhs_signature:
        return True
    if lhs_shape == rhs_shape:
        return False
    try:
        broadcast_shape = [int(v) for v in list(np.broadcast_shapes(tuple(lhs_shape), tuple(rhs_shape)))]
        if output_tensor is None:
            return False
        output_shape = [int(v) for v in list(output_tensor.shape)]
        if broadcast_shape != output_shape:
            return True
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else list(output_shape)
        )
        return (
            lhs_signature != lhs_shape
            or rhs_signature != rhs_shape
            or output_signature != output_shape
        )
    except Exception:
        return True
    return bool(
        (isinstance(lhs_tensor.data, np.ndarray) and len(lhs_shape) > 1)
        or (isinstance(rhs_tensor.data, np.ndarray) and len(rhs_shape) > 1)
    )

def _preferred_binary_alignment_anchor_for_codegen(
    *,
    model_ir: ModelIR,
    lhs_name: str,
    rhs_name: str,
    output_name: str,
) -> Optional[str]:
    lhs_tensor = model_ir.tensors.get(str(lhs_name), None)
    rhs_tensor = model_ir.tensors.get(str(rhs_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if lhs_tensor is None or rhs_tensor is None or output_tensor is None:
        return None
    lhs_signature = (
        [int(v) for v in list(lhs_tensor.shape_signature)]
        if lhs_tensor.shape_signature is not None
        else [int(v) for v in list(lhs_tensor.shape)]
    )
    rhs_signature = (
        [int(v) for v in list(rhs_tensor.shape_signature)]
        if rhs_tensor.shape_signature is not None
        else [int(v) for v in list(rhs_tensor.shape)]
    )
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_tensor.shape)]
    )
    if len(lhs_signature) != len(rhs_signature) or len(lhs_signature) != len(output_signature):
        return None
    if lhs_signature == output_signature and rhs_signature != output_signature:
        return "lhs"
    if rhs_signature == output_signature and lhs_signature != output_signature:
        return "rhs"
    lhs_dynamic_dims = sum(1 for dim in lhs_signature if int(dim) <= 0)
    rhs_dynamic_dims = sum(1 for dim in rhs_signature if int(dim) <= 0)
    if lhs_dynamic_dims > rhs_dynamic_dims:
        return "lhs"
    if rhs_dynamic_dims > lhs_dynamic_dims:
        return "rhs"
    return None

def _all_consumers_are_channel_first_binary_ops_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    direct_codegen_binary_functions: Collection[str],
    can_emit_channel_first_binary_op_fn: Callable[[OperatorIR], bool],
    output_name: str,
) -> bool:
    consumer_indices = consumer_index.get(str(output_name), [])
    if len(consumer_indices) == 0:
        return False
    for consumer_idx in consumer_indices:
        consumer_op = model_ir.operators[int(consumer_idx)]
        if str(consumer_op.op_type) not in direct_codegen_binary_functions:
            return False
        if str(output_name) not in {str(name) for name in list(consumer_op.inputs)}:
            return False
        if not can_emit_channel_first_binary_op_fn(consumer_op):
            return False
    return True

def _can_omit_materialized_channel_last_alias_recursive_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    direct_codegen_unary_expressions: Collection[str],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    channel_first_reduction_plan_fn: Callable[[OperatorIR, str], Optional[Tuple[str, List[int]]]],
    can_emit_channel_first_shape_preserving_unary_op_fn: Callable[[OperatorIR], bool],
    can_emit_channel_first_binary_op_fn: Callable[[OperatorIR], bool],
    can_resolve_channel_first_expr_statically_fn: Callable[[str], bool],
    conv2d_input_pre_permute_fn: Callable[..., Optional[List[int]]],
    output_name: str,
    seen_names: Set[str],
) -> bool:
    if str(output_name) in seen_names:
        return True
    if str(output_name) in {str(name) for name in list(model_ir.outputs)}:
        return False
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if output_tensor is None:
        return False
    output_rank = len(list(output_tensor.shape))
    expected_input_bridge_perm = _perm_cl_to_cf(output_rank)
    if expected_input_bridge_perm is None:
        return False
    consumer_indices = consumer_index.get(str(output_name), [])
    if len(consumer_indices) == 0:
        return True
    next_seen_names = set(seen_names)
    next_seen_names.add(str(output_name))
    for consumer_idx in consumer_indices:
        consumer_op = model_ir.operators[int(consumer_idx)]
        consumer_type = str(consumer_op.op_type)
        if consumer_type == "TRANSPOSE":
            transpose_perm = _read_transpose_perm(model_ir, consumer_op)
            if (
                len(consumer_op.inputs) < 1
                or str(consumer_op.inputs[0]) != str(output_name)
                or list(transpose_perm or []) != list(expected_input_bridge_perm)
            ):
                return False
            continue
        if consumer_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            if consumer_type == "DEPTHWISE_CONV_2D":
                return False
            if len(consumer_op.inputs) < 2 or str(consumer_op.inputs[0]) != str(output_name):
                return False
            input_pre_permute = conv2d_input_pre_permute_fn(
                tensor_shape_list_fn(str(consumer_op.inputs[0])),
                tensor_shape_list_fn(str(consumer_op.outputs[0])),
                tensor_shape_list_fn(str(consumer_op.inputs[1])),
                consumer_op.options,
                input_logical_layout=normalize_logical_layout(output_tensor.logical_layout),
                output_logical_layout=normalize_logical_layout(
                    model_ir.tensors[str(consumer_op.outputs[0])].logical_layout
                ),
                depthwise=(consumer_type == "DEPTHWISE_CONV_2D"),
            )
            if list(input_pre_permute or []) != list(expected_input_bridge_perm):
                return False
            continue
        if consumer_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            if channel_first_reduction_plan_fn(consumer_op, str(output_name)) is None:
                return False
            continue
        if consumer_type in direct_codegen_unary_expressions:
            if not can_emit_channel_first_shape_preserving_unary_op_fn(consumer_op):
                return False
            if len(consumer_op.inputs) != 1 or len(consumer_op.outputs) != 1:
                return False
            if str(consumer_op.inputs[0]) != str(output_name):
                return False
            consumer_output_name = str(consumer_op.outputs[0])
            consumer_output_tensor = model_ir.tensors.get(consumer_output_name, None)
            if consumer_output_tensor is None or len(list(consumer_output_tensor.shape)) != output_rank:
                return False
            current_shape = tensor_shape_list_fn(str(output_name))
            consumer_output_shape = tensor_shape_list_fn(consumer_output_name)
            if not _shape_lists_equal_relaxed(current_shape, consumer_output_shape):
                return False
            if not _can_omit_materialized_channel_last_alias_recursive_for_codegen(
                model_ir=model_ir,
                consumer_index=consumer_index,
                direct_codegen_unary_expressions=direct_codegen_unary_expressions,
                tensor_shape_list_fn=tensor_shape_list_fn,
                channel_first_reduction_plan_fn=channel_first_reduction_plan_fn,
                can_emit_channel_first_shape_preserving_unary_op_fn=can_emit_channel_first_shape_preserving_unary_op_fn,
                can_emit_channel_first_binary_op_fn=can_emit_channel_first_binary_op_fn,
                can_resolve_channel_first_expr_statically_fn=can_resolve_channel_first_expr_statically_fn,
                conv2d_input_pre_permute_fn=conv2d_input_pre_permute_fn,
                output_name=consumer_output_name,
                seen_names=next_seen_names,
            ):
                return False
            continue
        if consumer_type in {"ADD", "DIV", "MAXIMUM", "MINIMUM", "MUL", "SUB"}:
            if not can_emit_channel_first_binary_op_fn(consumer_op):
                return False
            if len(consumer_op.inputs) != 2 or len(consumer_op.outputs) != 1:
                return False
            output_name_set = {str(name) for name in list(consumer_op.inputs)}
            if str(output_name) not in output_name_set:
                return False
            consumer_output_name = str(consumer_op.outputs[0])
            consumer_output_tensor = model_ir.tensors.get(consumer_output_name, None)
            if (
                consumer_output_tensor is None
                or len(list(consumer_output_tensor.shape)) != output_rank
                or not is_channel_last_logical_layout(
                    normalize_logical_layout(consumer_output_tensor.logical_layout)
                )
            ):
                return False
            input_names = [str(name) for name in list(consumer_op.inputs)]
            dynamic_input_names = [
                input_name
                for input_name in input_names
                if (
                    model_ir.tensors.get(input_name, None) is not None
                    and model_ir.tensors[input_name].data is None
                )
            ]
            if len(dynamic_input_names) == 0:
                return False
            current_shape = tensor_shape_list_fn(str(output_name))
            consumer_output_shape = tensor_shape_list_fn(consumer_output_name)
            if not _shape_can_broadcast_to_target_relaxed(current_shape, consumer_output_shape):
                return False
            broadcast_shape: Optional[List[int]] = None
            for input_name in dynamic_input_names:
                input_tensor = model_ir.tensors.get(input_name, None)
                if input_tensor is None:
                    return False
                if not can_resolve_channel_first_expr_statically_fn(input_name):
                    return False
                input_shape = tensor_shape_list_fn(input_name)
                if input_shape is None or len(list(input_shape)) != output_rank:
                    return False
                if not _shape_can_broadcast_to_target_relaxed(input_shape, consumer_output_shape):
                    return False
                broadcast_shape = (
                    list(input_shape)
                    if broadcast_shape is None
                    else _broadcast_shapes_relaxed(broadcast_shape, input_shape)
                )
                if broadcast_shape is None:
                    return False
            if not _shape_lists_equal_relaxed(broadcast_shape, consumer_output_shape):
                return False
            if not _can_omit_materialized_channel_last_alias_recursive_for_codegen(
                model_ir=model_ir,
                consumer_index=consumer_index,
                direct_codegen_unary_expressions=direct_codegen_unary_expressions,
                tensor_shape_list_fn=tensor_shape_list_fn,
                channel_first_reduction_plan_fn=channel_first_reduction_plan_fn,
                can_emit_channel_first_shape_preserving_unary_op_fn=can_emit_channel_first_shape_preserving_unary_op_fn,
                can_emit_channel_first_binary_op_fn=can_emit_channel_first_binary_op_fn,
                can_resolve_channel_first_expr_statically_fn=can_resolve_channel_first_expr_statically_fn,
                conv2d_input_pre_permute_fn=conv2d_input_pre_permute_fn,
                output_name=consumer_output_name,
                seen_names=next_seen_names,
            ):
                return False
            continue
        return False
    return True

def _channel_first_binary_input_expr_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    channel_first_constant_buffer_alias_exprs: Dict[str, str],
    permuted_constant_buffer_alias_exprs: Dict[Tuple[str, Tuple[int, ...]], str],
    scalar_literal_expr_fn: Callable[[str], Optional[str]],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    tensor_expr_fn: Callable[[str], str],
    channel_first_passthrough_input_expr_fn: Callable[[str], Optional[str]],
    tensor_name: str,
    other_tensor_name: str,
) -> Optional[str]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    scalar_literal = scalar_literal_expr_fn(str(tensor_name))
    if scalar_literal is not None:
        return scalar_literal
    if isinstance(tensor.data, np.ndarray):
        tensor_shape = tensor_shape_list_fn(str(tensor_name))
        other_shape = tensor_shape_list_fn(str(other_tensor_name))
        tensor_layout = normalize_logical_layout(tensor.logical_layout)
        rank = len(list(tensor.shape))
        if (
            tensor_shape is not None
            and other_shape is not None
            and len(tensor_shape) == len(other_shape)
            and len(tensor_shape) in {3, 4, 5}
        ):
            other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
            other_target_shape = list(other_shape)
            constant_channel_dim = (
                int(tensor_shape[3])
                if rank == 4 and len(tensor_shape) == 4 and int(tensor_shape[3]) > 0
                else None
            )
            if other_tensor is not None:
                other_layout = normalize_logical_layout(other_tensor.logical_layout)
                other_perm_to_cf = None
                if (
                    constant_channel_dim is not None
                    and len(other_shape) == 4
                    and int(other_shape[-1]) == int(constant_channel_dim)
                    and int(other_shape[1]) != int(constant_channel_dim)
                ):
                    other_perm_to_cf = _perm_cl_to_cf(len(other_shape))
                elif (
                    is_channel_last_logical_layout(other_layout)
                    or (
                        len(other_shape) == 4
                        and "_nhwc" in str(other_tensor_name)
                    )
                ):
                    other_perm_to_cf = _perm_cl_to_cf(len(other_shape))
                if (
                    other_perm_to_cf is not None
                    and (
                        is_channel_first_logical_layout(other_layout)
                        or str(other_tensor_name) in channel_first_tensor_expr_aliases
                    )
                ):
                    permuted_other_shape = _permute_shape(other_shape, other_perm_to_cf)
                    if permuted_other_shape is not None:
                        other_target_shape = permuted_other_shape
            buffer_alias_expr = channel_first_constant_buffer_alias_exprs.get(str(tensor_name), None)
            if (
                rank == 4
                and buffer_alias_expr is not None
                and tensor_shape == [1, 1, 1, int(tensor_shape[3])]
            ):
                channel_first_constant_shape = [1, int(tensor_shape[3]), 1, 1]
                if _shape_can_broadcast_to_target_relaxed(
                    channel_first_constant_shape,
                    other_target_shape,
                ):
                    return str(buffer_alias_expr)
            perm_to_cf = _perm_cl_to_cf(rank) if is_channel_last_logical_layout(tensor_layout) else None
            if perm_to_cf is not None:
                permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
                base_expr = tensor_expr_fn(str(tensor_name))
                permuted_alias_expr = permuted_constant_buffer_alias_exprs.get(
                    (str(tensor_name), tuple(int(v) for v in list(perm_to_cf)))
                )
                if _shape_can_broadcast_to_target_relaxed(permuted_shape, other_target_shape):
                    singleton_spatial = (
                        permuted_shape is not None
                        and len(permuted_shape) >= 3
                        and all(int(v) == 1 for v in list(permuted_shape[2:]))
                    )
                    resolved_permuted_shape = permuted_shape if singleton_spatial else None
                    if (
                        resolved_permuted_shape is not None
                        and tensor_shape[0] == resolved_permuted_shape[0]
                    ):
                        if buffer_alias_expr is not None:
                            return str(buffer_alias_expr)
                        return f"torch.reshape({base_expr}, {repr(resolved_permuted_shape)})"
                    if permuted_alias_expr is not None:
                        return str(permuted_alias_expr)
                    return (
                        f"{base_expr}.permute("
                        f"{', '.join(str(int(v)) for v in perm_to_cf)}).contiguous()"
                    )
        return None
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    passthrough_expr = channel_first_passthrough_input_expr_fn(str(tensor_name))
    if passthrough_expr is not None:
        return str(passthrough_expr)
    other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
    if other_tensor is None:
        return None
    if (
        len(list(tensor.shape)) == len(list(other_tensor.shape))
        and _shape_lists_equal(tensor_shape_list_fn(str(tensor_name)), tensor_shape_list_fn(str(other_tensor_name)))
    ):
        preferred_perm: Optional[List[int]] = None
        rank = len(list(tensor.shape))
        if is_channel_last_logical_layout(tensor_layout):
            preferred_perm = _perm_cl_to_cf(rank)
        if preferred_perm is not None:
            return (
                f"{tensor_expr_fn(str(tensor_name))}.permute("
                f"{', '.join(str(int(v)) for v in preferred_perm)}).contiguous()"
            )
    return None

def _can_emit_channel_first_binary_op_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    channel_first_shape_for_tensor_fn: Callable[[str], Optional[List[int]]],
    scalar_literal_expr_fn: Callable[[str], Optional[str]],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    channel_first_binary_input_expr_fn: Callable[[str, str], Optional[str]],
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    if op_type not in {"ADD", "DIV", "MAXIMUM", "MINIMUM", "MUL", "SUB"}:
        return False
    if len(op.inputs) != 2 or len(op.outputs) != 1:
        return False
    output_name = str(op.outputs[0])
    output_tensor = model_ir.tensors.get(output_name, None)
    if output_tensor is None:
        return False
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    can_skip_materialized_output = can_omit_materialized_channel_last_alias_fn(output_name)
    if not (
        is_channel_last_logical_layout(output_layout)
        or is_channel_first_logical_layout(output_layout)
        or (output_layout == LOGICAL_LAYOUT_UNKNOWN and can_skip_materialized_output)
    ):
        return False
    output_shape = channel_first_shape_for_tensor_fn(output_name)
    output_rank = len(list(output_tensor.shape))
    if output_shape is None or output_rank not in {3, 4, 5}:
        return False
    dynamic_input_names: List[str] = []
    broadcast_shape: Optional[List[int]] = None
    for input_name in op.inputs:
        input_tensor = model_ir.tensors.get(str(input_name), None)
        if input_tensor is None:
            return False
        if input_tensor.data is None:
            dynamic_input_names.append(str(input_name))
            input_shape = channel_first_shape_for_tensor_fn(str(input_name))
            if (
                input_shape is None
                or len(list(input_shape)) != output_rank
                or not _shape_can_broadcast_to_target_relaxed(input_shape, output_shape)
            ):
                return False
            broadcast_shape = (
                list(input_shape)
                if broadcast_shape is None
                else _broadcast_shapes_relaxed(broadcast_shape, input_shape)
            )
            if broadcast_shape is None:
                return False
            other_input_name = str(op.inputs[1]) if str(input_name) == str(op.inputs[0]) else str(op.inputs[0])
            if channel_first_binary_input_expr_fn(str(input_name), other_input_name) is None:
                return False
            continue
        if scalar_literal_expr_fn(str(input_name)) is not None:
            continue
        other_input_name = str(op.inputs[1]) if str(input_name) == str(op.inputs[0]) else str(op.inputs[0])
        input_cf_expr = channel_first_binary_input_expr_fn(str(input_name), other_input_name)
        if input_cf_expr is None:
            return False
        input_shape = channel_first_shape_for_tensor_fn(str(input_name))
        raw_input_shape = tensor_shape_list_fn(str(input_name))
        if (
            raw_input_shape is not None
            and len(raw_input_shape) == output_rank == 4
            and [int(v) for v in list(raw_input_shape[:3])] == [1, 1, 1]
            and int(raw_input_shape[3]) > 0
        ):
            input_shape = [1, int(raw_input_shape[3]), 1, 1]
        if (
            input_shape is None
            or len(list(input_shape)) != output_rank
            or not _shape_can_broadcast_to_target_relaxed(input_shape, output_shape)
        ):
            return False
        broadcast_shape = (
            list(input_shape)
            if broadcast_shape is None
            else _broadcast_shapes_relaxed(broadcast_shape, input_shape)
        )
        if broadcast_shape is None:
            return False
    if len(dynamic_input_names) == 0:
        return False
    if not _shape_lists_equal_relaxed(broadcast_shape, output_shape):
        return False
    return True

def _binary_operand_expr_for_codegen(
    *,
    model_ir: ModelIR,
    binary_constant_buffer_alias_exprs: Dict[Tuple[str, str], str],
    channel_first_constant_buffer_alias_exprs: Dict[str, str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_expr_fn: Callable[[str], str],
    channel_first_rank4_constant_buffer_alias_shape_fn: Callable[[str], Optional[List[int]]],
    channel_first_shape_for_tensor_fn: Callable[[str], Optional[List[int]]],
    constant_permute_for_broadcast_fn: Callable[[str, str], Optional[List[int]]],
    permuted_constant_expr_for_tensor_name_fn: Callable[[str, Sequence[int]], Optional[str]],
    tensor_name: str,
    other_tensor_name: str,
) -> str:
    def _has_unambiguous_channel_first_shape(shape: Sequence[int]) -> bool:
        return (
            len(list(shape)) in {3, 4, 5}
            and len(list(shape)) >= 2
            and int(shape[1]) != int(shape[-1])
        )

    alias_expr = binary_constant_buffer_alias_exprs.get((str(tensor_name), str(other_tensor_name)), None)
    expr = tensor_expr_fn(tensor_name)
    other_expr = tensor_expr_fn(other_tensor_name)
    tensor = model_ir.tensors.get(str(tensor_name), None)
    other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
    if tensor is None or other_tensor is None:
        if alias_expr is not None:
            return str(alias_expr)
        return expr
    channel_first_alias_expr = channel_first_constant_buffer_alias_exprs.get(str(tensor_name), None)
    channel_first_alias_shape = channel_first_rank4_constant_buffer_alias_shape_fn(str(tensor_name))
    other_layout = normalize_logical_layout(other_tensor.logical_layout)
    other_prefers_channel_first = (
        str(other_tensor_name) in channel_first_tensor_expr_aliases
        or (
            is_channel_first_logical_layout(other_layout)
            and _has_unambiguous_channel_first_shape(other_tensor.shape)
        )
        or str(other_expr).endswith("_cf")
        or (
            ".permute(" in str(other_expr)
            and ".contiguous()" in str(other_expr)
        )
    )
    if (
        channel_first_alias_expr is not None
        and channel_first_alias_shape is not None
        and other_prefers_channel_first
    ):
        other_cf_shape = channel_first_shape_for_tensor_fn(str(other_tensor_name))
        broadcast_shape = _broadcast_shapes_relaxed(
            channel_first_alias_shape,
            other_cf_shape,
        )
        if _shape_lists_equal_relaxed(broadcast_shape, other_cf_shape):
            return str(channel_first_alias_expr)
    if alias_expr is not None:
        return str(alias_expr)
    if not isinstance(tensor.data, np.ndarray):
        return expr
    tensor_shape = [int(v) for v in list(tensor.shape)]
    other_shape = [int(v) for v in list(other_tensor.shape)]
    tensor_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in tensor_shape]
    other_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in other_shape]
    constant_width = max([int(dim) for dim in tensor_shape], default=0)
    if len(tensor_shape) < len(other_shape):
        target_axis: Optional[int] = None
        other_layout = normalize_logical_layout(other_tensor.logical_layout)
        if (
            is_channel_first_logical_layout(other_layout)
            and len(other_shape) >= 2
            and int(other_shape[1]) == constant_width
        ):
            target_axis = 1
        elif (
            is_channel_last_logical_layout(other_layout)
            and len(other_shape) >= 1
            and int(other_shape[-1]) == constant_width
        ):
            target_axis = len(other_shape) - 1
        else:
            matching_axes = [
                int(axis)
                for axis, dim in enumerate(other_shape)
                if int(axis) != 0 and int(dim) == constant_width
            ]
            if len(matching_axes) == 1:
                target_axis = int(matching_axes[0])
        if target_axis is not None and len(tensor_shape) >= 1:
            batch_dim = int(tensor_shape[0])
            if batch_dim in {1, int(other_shape[0])}:
                non_batch_non_singleton_axes = [
                    int(axis)
                    for axis, dim in enumerate(tensor_shape[1:], start=1)
                    if int(dim) > 1
                ]
                if len(non_batch_non_singleton_axes) == 1:
                    source_axis = int(non_batch_non_singleton_axes[0])
                    if int(tensor_shape[source_axis]) == constant_width:
                        other_non_target_axes = [
                            int(axis)
                            for axis, dim in enumerate(other_shape)
                            if int(axis) not in {0, int(target_axis)} and int(dim) > 1
                        ]
                        if len(other_non_target_axes) > 0:
                            if (
                                len(tensor_shape) == 1
                                and int(target_axis) == len(other_shape) - 1
                            ):
                                return expr
                            reshape_dims = [1 for _ in other_shape]
                            reshape_dims[0] = int(batch_dim)
                            reshape_dims[int(target_axis)] = int(constant_width)
                            return f"{expr}.reshape({repr(reshape_dims)})"
    if len(tensor_shape) != 1 or len(other_shape) < 2:
        if len(tensor_shape) == len(other_shape) and len(tensor_shape) > 1:
            selected_perm = constant_permute_for_broadcast_fn(str(tensor_name), str(other_tensor_name))
            if selected_perm is not None:
                selected_alias_expr = permuted_constant_expr_for_tensor_name_fn(
                    str(tensor_name),
                    selected_perm,
                )
                if selected_alias_expr is not None:
                    return str(selected_alias_expr)
                return f"{expr}.permute(*{repr(tuple(int(v) for v in selected_perm))}).contiguous()"
            try:
                np.broadcast_shapes(tuple(tensor_broadcast_shape), tuple(other_broadcast_shape))
                return expr
            except Exception:
                pass
            preferred_perm: Optional[Tuple[int, ...]] = None
            tensor_layout = normalize_logical_layout(tensor.logical_layout)
            if tensor_layout == "NCHW" and len(tensor_shape) == 4:
                preferred_perm = (0, 2, 3, 1)
            elif tensor_layout == "NCDHW" and len(tensor_shape) == 5:
                preferred_perm = (0, 2, 3, 4, 1)
            elif tensor_layout == "NCW" and len(tensor_shape) == 3:
                preferred_perm = (0, 2, 1)
            if preferred_perm is not None:
                preferred_shape = [int(tensor_shape[int(idx)]) for idx in preferred_perm]
                preferred_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in preferred_shape]
                try:
                    np.broadcast_shapes(tuple(preferred_broadcast_shape), tuple(other_broadcast_shape))
                    preferred_alias_expr = permuted_constant_expr_for_tensor_name_fn(
                        str(tensor_name),
                        preferred_perm,
                    )
                    if preferred_alias_expr is not None:
                        return str(preferred_alias_expr)
                    return f"{expr}.permute(*{repr(tuple(int(v) for v in preferred_perm))}).contiguous()"
                except Exception:
                    pass
            import itertools

            for generic_perm in itertools.permutations(range(len(tensor_shape))):
                if list(generic_perm) == list(range(len(tensor_shape))):
                    continue
                permuted_shape = [int(tensor_shape[int(idx)]) for idx in generic_perm]
                permuted_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in permuted_shape]
                try:
                    np.broadcast_shapes(tuple(permuted_broadcast_shape), tuple(other_broadcast_shape))
                    generic_alias_expr = permuted_constant_expr_for_tensor_name_fn(
                        str(tensor_name),
                        generic_perm,
                    )
                    if generic_alias_expr is not None:
                        return str(generic_alias_expr)
                    return f"{expr}.permute(*{repr(tuple(int(v) for v in generic_perm))}).contiguous()"
                except Exception:
                    continue
        return expr
    constant_width = int(tensor_shape[0])
    if constant_width <= 0:
        return expr
    target_axis: Optional[int] = None
    other_layout = normalize_logical_layout(other_tensor.logical_layout)
    if (
        is_channel_first_logical_layout(other_layout)
        and len(other_shape) >= 2
        and int(other_shape[1]) == constant_width
    ):
        target_axis = 1
    elif (
        is_channel_last_logical_layout(other_layout)
        and int(other_shape[-1]) == constant_width
    ):
        target_axis = len(other_shape) - 1
    else:
        matching_axes = [
            int(axis)
            for axis, dim in enumerate(other_shape)
            if int(axis) != 0 and int(dim) == constant_width
        ]
        if len(matching_axes) == 1:
            target_axis = int(matching_axes[0])
    if target_axis is None:
        return expr
    if len(tensor_shape) == 1 and int(target_axis) == len(other_shape) - 1:
        return expr
    reshape_dims = [1 for _ in other_shape]
    reshape_dims[int(target_axis)] = constant_width
    if reshape_dims == tensor_shape:
        return expr
    return f"{expr}.reshape({repr(reshape_dims)})"

def _emit_native_direct_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_index: int,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    op_module_attr_names: Dict[int, str],
    fused_module_specs: Dict[int, Dict[str, Any]],
    conv_module_pad_specs: Dict[int, Optional[List[int]]],
    tensor_var_names: Dict[str, str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    tensor_expr_for_channel_first_bridge_fn: Callable[[str, Sequence[int]], Optional[str]],
    all_consumers_are_channel_first_binary_ops_fn: Callable[[str], bool],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    derived_local_var_name_fn: Callable[[str, str], str],
    emit_module_output_expr_fn: Callable[..., str],
    target_shape_literal_fn: Callable[[str], str],
    conv2d_input_pre_permute_fn: Callable[..., Optional[List[int]]],
    can_emit_direct_module_call_fn: Callable[[OperatorIR], bool],
    activation_lines_fn: Callable[[str, str], List[str]],
    emit_maybe_aligned_expr_fn: Callable[..., str],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    should_skip_align_for_shape_preserving_unary_fn: Callable[[str, str], bool],
) -> bool:
    op_type = str(op.op_type)
    if op_type not in _DIRECT_CODEGEN_MODULE_OP_TYPES:
        return False
    attr_name = op_module_attr_names[int(op_index)]
    fused_module_spec = fused_module_specs.get(int(op_index), None)
    if op_type == "UNIDIRECTIONAL_SEQUENCE_RNN":
        x_expr = tensor_expr_fn(str(op.inputs[0]))
        h0_name = _sequence_lstm_input_name(op, 4)
        state_arg = tensor_expr_fn(h0_name) if h0_name != "" else "None"
        forward_lines.append(
            f"{output_vars[0]} = _align_tensor_to_target_shape("
            f"self.{attr_name}({x_expr}, {state_arg}), "
            f"{output_target_shape})"
        )
        runtime_imports.add("_align_tensor_to_target_shape")
        return True
    if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        x_expr = tensor_expr_fn(str(op.inputs[0]))
        index_spec = _sequence_lstm_index_spec(op)
        if index_spec is None:
            raise ModelIRPyTorchExportError(
                "Native PyTorch-like model.py codegen could not resolve UNIDIRECTIONAL_SEQUENCE_LSTM state layout."
            )
        state_indices = list(index_spec["state_indices"])
        h0_name = _sequence_lstm_input_name(op, state_indices[0])
        c0_name = _sequence_lstm_input_name(op, state_indices[1])
        state_args = [
            tensor_expr_fn(h0_name) if h0_name != "" else "None",
            tensor_expr_fn(c0_name) if c0_name != "" else "None",
        ]
        forward_lines.append(
            f"{output_vars[0]} = _align_tensor_to_target_shape("
            f"self.{attr_name}({x_expr}, {', '.join(state_args)}), "
            f"{output_target_shape})"
        )
        runtime_imports.add("_align_tensor_to_target_shape")
        return True
    if op_type == "BIDIRECTIONAL_SEQUENCE_LSTM":
        x_expr = tensor_expr_fn(str(op.inputs[0]))
        index_spec = _sequence_lstm_index_spec(op)
        if index_spec is None:
            raise ModelIRPyTorchExportError(
                "Native PyTorch-like model.py codegen could not resolve BIDIRECTIONAL_SEQUENCE_LSTM state layout."
            )
        state_indices = list(index_spec["state_indices"])
        fw_h0_name = _sequence_lstm_input_name(op, state_indices[0])
        fw_c0_name = _sequence_lstm_input_name(op, state_indices[1])
        bw_h0_name = _sequence_lstm_input_name(op, state_indices[2])
        bw_c0_name = _sequence_lstm_input_name(op, state_indices[3])
        state_args = [
            tensor_expr_fn(fw_h0_name) if fw_h0_name != "" else "None",
            tensor_expr_fn(fw_c0_name) if fw_c0_name != "" else "None",
            tensor_expr_fn(bw_h0_name) if bw_h0_name != "" else "None",
            tensor_expr_fn(bw_c0_name) if bw_c0_name != "" else "None",
        ]
        forward_lines.append(
            f"{output_vars[0]} = _align_tensor_to_target_shape("
            f"self.{attr_name}({x_expr}, {', '.join(state_args)}), "
            f"{output_target_shape})"
        )
        runtime_imports.add("_align_tensor_to_target_shape")
        return True
    if fused_module_spec is not None:
        output_name = str(fused_module_spec["output_name"])
        output_var = tensor_var_names[output_name]
        fused_input_expr = tensor_expr_fn(str(fused_module_spec["input_name"]))
        input_pre_permute = fused_module_spec.get("input_pre_permute", None)
        fused_input_tensor = model_ir.tensors.get(str(fused_module_spec["input_name"]), None)
        output_tensor = model_ir.tensors.get(output_name, None)
        fallback_channel_last_conv_bridge = False
        if isinstance(input_pre_permute, list) and len(input_pre_permute) == 4:
            folded_channel_first_expr = tensor_expr_for_channel_first_bridge_fn(
                str(fused_module_spec["input_name"]),
                input_pre_permute,
            )
            if folded_channel_first_expr is not None:
                fused_input_expr = str(folded_channel_first_expr)
            else:
                fused_input_expr = (
                    f"{fused_input_expr}.permute({', '.join(str(int(v)) for v in input_pre_permute)}).contiguous()"
                )
        elif (
            op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
            and fused_input_tensor is not None
            and output_tensor is not None
            and is_channel_last_logical_layout(
                normalize_logical_layout(fused_input_tensor.logical_layout)
            )
            and is_channel_last_logical_layout(
                normalize_logical_layout(output_tensor.logical_layout)
            )
        ):
            fused_input_expr = f"{fused_input_expr}.permute(0, 3, 1, 2).contiguous()"
            fallback_channel_last_conv_bridge = True
        raw_output_layout = LOGICAL_LAYOUT_UNKNOWN
        if output_tensor is not None:
            output_rank = len(list(output_tensor.shape))
            if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV"}:
                raw_output_layout = channel_first_logical_layout(output_rank)
            elif op_type == "CONV_3D":
                raw_output_layout = channel_first_logical_layout(output_rank)
        fused_module_expr = f"self.{attr_name}({fused_input_expr})"
        if fallback_channel_last_conv_bridge:
            fused_module_expr = f"{fused_module_expr}.permute(0, 2, 3, 1).contiguous()"
            raw_output_layout = normalize_logical_layout(
                output_tensor.logical_layout if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            )
        output_layout = normalize_logical_layout(
            output_tensor.logical_layout if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        )
        normalized_raw_output_layout = normalize_logical_layout(raw_output_layout)
        if (
            output_tensor is not None
            and is_channel_first_logical_layout(normalized_raw_output_layout)
        ):
            output_rank = len(list(output_tensor.shape))
            if output_rank in {3, 4, 5}:
                raw_output_var = derived_local_var_name_fn(f"{output_var}_cf", "t")
                channel_first_tensor_expr_aliases[str(output_name)] = raw_output_var
                forward_lines.append(f"{raw_output_var} = {fused_module_expr}")
                if output_layout == normalized_raw_output_layout:
                    if output_name in model_ir.outputs:
                        weight_tensor = model_ir.tensors.get(str(op.inputs[1]), None) if len(op.inputs) >= 2 else None
                        if (
                            output_rank == 4
                            and weight_tensor is not None
                            and len(list(weight_tensor.shape)) >= 1
                            and int(output_tensor.shape[-1]) == int(weight_tensor.shape[0])
                            and int(output_tensor.shape[1]) != int(weight_tensor.shape[0])
                        ):
                            runtime_imports.add("_align_tensor_to_target_shape")
                            forward_lines.append(
                                f"{output_var} = _align_tensor_to_target_shape("
                                f"{raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                                f"{target_shape_literal_fn(output_name)})"
                            )
                        else:
                            forward_lines.append(f"{output_var} = {raw_output_var}")
                    return True
                if is_channel_last_logical_layout(output_layout):
                    if (
                        all_consumers_are_channel_first_binary_ops_fn(output_name)
                        or can_omit_materialized_channel_last_alias_fn(output_name)
                    ):
                        return True
                    perm_to_output = logical_layout_permutation(
                        source_layout=normalized_raw_output_layout,
                        target_layout=output_layout,
                    )
                    if perm_to_output is None:
                        raise ModelIRPyTorchExportError(
                            "Native PyTorch-like model.py codegen could not derive a layout bridge "
                            f"for fused module output. output={output_name} raw_layout={normalized_raw_output_layout} "
                            f"target_layout={output_layout}"
                        )
                    runtime_imports.add("_align_tensor_to_target_shape")
                    forward_lines.append(
                        f"{output_var} = _align_tensor_to_target_shape("
                        f"{raw_output_var}.permute({', '.join(str(int(v)) for v in perm_to_output)}).contiguous(), "
                        f"{target_shape_literal_fn(output_name)})"
                    )
                    return True
                forward_lines.append(
                    f"{output_var} = {emit_module_output_expr_fn(output_name=output_name, expr=raw_output_var, raw_output_layout=raw_output_layout)}"
                )
                return True
        channel_first_tensor_expr_aliases.pop(str(output_name), None)
        forward_lines.append(
            f"{output_var} = {emit_module_output_expr_fn(output_name=output_name, expr=fused_module_expr, raw_output_layout=raw_output_layout)}"
        )
        return True
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
        conv_input_name = str(op.inputs[0])
        conv_input_expr = tensor_expr_fn(conv_input_name)
        output_name = str(outputs[0])
        output_tensor = model_ir.tensors.get(output_name, None)
        raw_output_layout = (
            channel_first_logical_layout(len(list(output_tensor.shape)))
            if output_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        output_layout = normalize_logical_layout(
            output_tensor.logical_layout if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        )
        use_channel_first_alias = (
            output_tensor is not None
            and len(list(output_tensor.shape)) in {3, 4, 5}
            and is_channel_first_logical_layout(output_layout)
        )
        raw_output_var = (
            derived_local_var_name_fn(f"{output_vars[0]}_cf", "t")
            if use_channel_first_alias
            else output_vars[0]
        )
        conv_input_tensor = model_ir.tensors.get(conv_input_name, None)
        conv_weight_tensor = model_ir.tensors.get(str(op.inputs[1]), None) if len(op.inputs) >= 2 else None
        conv_input_layout = normalize_logical_layout(
            conv_input_tensor.logical_layout
            if conv_input_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        existing_channel_first_input_alias = channel_first_tensor_expr_aliases.get(conv_input_name, None)
        if (
            conv_input_tensor is not None
            and conv_weight_tensor is not None
            and len(list(conv_input_tensor.shape)) == 4
            and len(list(conv_weight_tensor.shape)) == 4
            and int(conv_input_tensor.shape[1]) == int(conv_weight_tensor.shape[1])
        ):
            input_pre_permute = None
        elif (
            conv_input_tensor is not None
            and len(list(conv_input_tensor.shape)) == 4
            and is_channel_first_logical_layout(conv_input_layout)
        ):
            input_pre_permute = None
        elif (
            existing_channel_first_input_alias is not None
            and conv_input_tensor is not None
            and conv_weight_tensor is not None
            and len(list(conv_input_tensor.shape)) == 4
            and len(list(conv_weight_tensor.shape)) == 4
            and int(conv_input_tensor.shape[1]) == 1
            and int(conv_input_tensor.shape[2]) == 1
            and int(conv_input_tensor.shape[3]) == int(conv_weight_tensor.shape[3])
        ):
            conv_input_expr = str(existing_channel_first_input_alias)
            input_pre_permute = None
        elif (
            conv_input_tensor is not None
            and conv_weight_tensor is not None
            and len(list(conv_input_tensor.shape)) == 4
            and len(list(conv_weight_tensor.shape)) == 4
            and int(conv_input_tensor.shape[1]) == 1
            and int(conv_input_tensor.shape[2]) == 1
            and int(conv_input_tensor.shape[3]) == int(conv_weight_tensor.shape[3])
        ):
            input_pre_permute = None
        else:
            input_pre_permute = conv2d_input_pre_permute_fn(
                tensor_shape_list_fn(str(op.inputs[0])),
                tensor_shape_list_fn(str(outputs[0])),
                tensor_shape_list_fn(str(op.inputs[1])),
                op.options,
                input_logical_layout=conv_input_layout,
                output_logical_layout=normalize_logical_layout(
                    model_ir.tensors[outputs[0]].logical_layout
                ),
                depthwise=(op_type == "DEPTHWISE_CONV_2D"),
            )
        if input_pre_permute is not None:
            folded_channel_first_expr = None
            if not (
                op_type == "DEPTHWISE_CONV_2D"
                and conv_input_tensor is not None
                and is_channel_last_logical_layout(conv_input_layout)
            ):
                folded_channel_first_expr = tensor_expr_for_channel_first_bridge_fn(
                    str(op.inputs[0]),
                    input_pre_permute,
                )
            if folded_channel_first_expr is not None:
                conv_input_expr = str(folded_channel_first_expr)
            else:
                conv_input_expr = (
                    f"{conv_input_expr}.permute({', '.join(str(int(v)) for v in input_pre_permute)}).contiguous()"
                )
        elif (
            op_type == "DEPTHWISE_CONV_2D"
            and conv_input_tensor is not None
            and is_channel_last_logical_layout(conv_input_layout)
            and existing_channel_first_input_alias is not None
        ):
            conv_input_expr = (
                f"{conv_input_expr}.permute(0, 2, 3, 1).contiguous()"
                ".permute(0, 3, 1, 2).contiguous()"
            )
        conv_pad_arg = conv_module_pad_specs.get(int(op_index), None)
        if conv_pad_arg is not None:
            conv_input_expr = f"F.pad({conv_input_expr}, {repr(conv_pad_arg)}, mode='constant', value=0.0)"
        if can_emit_direct_module_call_fn(op):
            if use_channel_first_alias:
                channel_first_tensor_expr_aliases[output_name] = raw_output_var
            else:
                channel_first_tensor_expr_aliases.pop(output_name, None)
            forward_lines.append(f"{raw_output_var} = self.{attr_name}({conv_input_expr})")
        else:
            runtime_imports.add("_apply_module_conv2d")
            forward_lines.append(
                f"{output_vars[0]} = _apply_module_conv2d(self.{attr_name}, {conv_input_expr}, target_shape={output_target_shape}, target_logical_layout={repr(normalize_logical_layout(model_ir.tensors[outputs[0]].logical_layout))}, fused='NONE')"
            )
            channel_first_tensor_expr_aliases.pop(output_name, None)
        forward_lines.extend(activation_lines_fn(output_vars[0], fused))
        if use_channel_first_alias:
            if fused != "NONE":
                forward_lines[-1] = forward_lines[-1].replace(f"{output_vars[0]} =", f"{raw_output_var} =", 1)
            if output_name in model_ir.outputs:
                public_output_expr = emit_module_output_expr_fn(
                    output_name=output_name,
                    expr=raw_output_var,
                    raw_output_layout=raw_output_layout,
                )
                weight_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if (
                    output_tensor is not None
                    and weight_tensor is not None
                    and len(list(output_tensor.shape)) == 4
                    and len(list(weight_tensor.shape)) >= 1
                    and int(output_tensor.shape[-1]) == int(weight_tensor.shape[0])
                    and int(output_tensor.shape[1]) != int(weight_tensor.shape[0])
                ):
                    runtime_imports.add("_align_tensor_to_target_shape")
                    public_output_expr = (
                        f"_align_tensor_to_target_shape({raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                        f"{target_shape_literal_fn(output_name)})"
                    )
                forward_lines.append(f"{output_vars[0]} = {public_output_expr}")
        return True
    if op_type == "TRANSPOSE_CONV":
        runtime_imports.add("_apply_module_transpose_conv2d")
        output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        fallback_shape = (
            [int(v) for v in np.asarray(output_shape_tensor.data).reshape(-1).tolist()]
            if output_shape_tensor is not None and isinstance(output_shape_tensor.data, np.ndarray)
            else [int(v) for v in list(model_ir.tensors[outputs[0]].shape)]
        )
        forward_lines.append(
            f"{output_vars[0]} = _apply_module_transpose_conv2d("
            f"{tensor_expr_fn(str(op.inputs[2]))}, self.{attr_name}.weight, self.{attr_name}.bias, "
            f"list(self.{attr_name}.stride), list(self.{attr_name}.padding), list(self.{attr_name}.dilation), "
            f"list(self.{attr_name}.output_padding), self.{attr_name}.groups, "
            f"target_shape={output_target_shape}, fallback_shape={repr(fallback_shape)}, "
            f"target_logical_layout={repr(normalize_logical_layout(model_ir.tensors[outputs[0]].logical_layout))}, fused='NONE')"
        )
        forward_lines.extend(activation_lines_fn(output_vars[0], fused))
        return True
    if op_type == "CONV_3D":
        if can_emit_direct_module_call_fn(op):
            forward_lines.append(f"{output_vars[0]} = self.{attr_name}({tensor_expr_fn(str(op.inputs[0]))})")
        else:
            runtime_imports.add("_apply_module_conv3d")
            forward_lines.append(
                f"{output_vars[0]} = _apply_module_conv3d(self.{attr_name}, {tensor_expr_fn(str(op.inputs[0]))}, target_shape={output_target_shape}, target_logical_layout={repr(normalize_logical_layout(model_ir.tensors[outputs[0]].logical_layout))}, fused='NONE')"
            )
        forward_lines.extend(activation_lines_fn(output_vars[0], fused))
        return True
    if op_type == "CONV_3D_TRANSPOSE":
        runtime_imports.add("_apply_module_transpose_conv3d")
        output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        fallback_shape = (
            [int(v) for v in np.asarray(output_shape_tensor.data).reshape(-1).tolist()]
            if output_shape_tensor is not None and isinstance(output_shape_tensor.data, np.ndarray)
            else [int(v) for v in list(model_ir.tensors[outputs[0]].shape)]
        )
        forward_lines.append(
            f"{output_vars[0]} = _apply_module_transpose_conv3d("
            f"{tensor_expr_fn(str(op.inputs[2]))}, self.{attr_name}.weight, self.{attr_name}.bias, "
            f"list(self.{attr_name}.stride), list(self.{attr_name}.padding), list(self.{attr_name}.dilation), "
            f"list(self.{attr_name}.output_padding), self.{attr_name}.groups, "
            f"target_shape={output_target_shape}, fallback_shape={repr(fallback_shape)}, "
            f"target_logical_layout={repr(normalize_logical_layout(model_ir.tensors[outputs[0]].logical_layout))}, fused='NONE')"
        )
        forward_lines.extend(activation_lines_fn(output_vars[0], fused))
        return True
    if op_type == "FULLY_CONNECTED":
        forward_lines.append(f"{output_vars[0]} = self.{attr_name}({tensor_expr_fn(str(op.inputs[0]))})")
        forward_lines.extend(activation_lines_fn(output_vars[0], fused))
        return True
    if op_type == "PRELU":
        prelu_input_name = str(op.inputs[0])
        prelu_input_expr = tensor_expr_fn(prelu_input_name)
        prelu_input_tensor = model_ir.tensors.get(prelu_input_name, None)
        prelu_output_tensor = model_ir.tensors.get(outputs[0], None)
        prelu_input_layout = (
            normalize_logical_layout(prelu_input_tensor.logical_layout)
            if prelu_input_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        prelu_output_layout = (
            normalize_logical_layout(prelu_output_tensor.logical_layout)
            if prelu_output_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        prelu_rank = len(list(prelu_input_tensor.shape)) if prelu_input_tensor is not None else 0
        prelu_num_parameters = 1
        prelu_weight_tensor = model_ir.tensors.get(str(op.inputs[1]), None) if len(op.inputs) >= 2 else None
        if prelu_weight_tensor is not None:
            if isinstance(prelu_weight_tensor.data, np.ndarray):
                prelu_num_parameters = max(1, int(np.asarray(prelu_weight_tensor.data).size))
            elif len(list(prelu_weight_tensor.shape)) > 0:
                shape_values = [int(v) for v in list(prelu_weight_tensor.shape) if int(v) > 0]
                if len(shape_values) > 0:
                    prelu_num_parameters = max(1, int(np.prod(shape_values, dtype=np.int64)))
        expr = f"self.{attr_name}({prelu_input_expr})"
        if (
            prelu_num_parameters > 1
            and prelu_rank in {3, 4, 5}
            and is_channel_last_logical_layout(prelu_input_layout)
        ):
            pre_perm = logical_layout_permutation(
                source_layout=prelu_input_layout,
                target_layout=channel_first_logical_layout(prelu_rank),
            )
            post_perm = logical_layout_permutation(
                source_layout=channel_first_logical_layout(prelu_rank),
                target_layout=prelu_output_layout if is_channel_last_logical_layout(prelu_output_layout) else prelu_input_layout,
            )
            if pre_perm is not None and post_perm is not None:
                permuted_input_expr = (
                    f"{prelu_input_expr}.permute({', '.join(str(int(v)) for v in pre_perm)}).contiguous()"
                )
                expr = (
                    f"self.{attr_name}({permuted_input_expr}).permute({', '.join(str(int(v)) for v in post_perm)}).contiguous()"
                )
        inferred_shape = tensor_shape_list_fn(str(op.inputs[0]))
        if should_skip_align_for_shape_preserving_unary_fn(str(op.inputs[0]), str(outputs[0])):
            forward_lines.append(f"{output_vars[0]} = {expr}")
        else:
            forward_lines.append(
                f"{output_vars[0]} = {emit_maybe_aligned_expr_fn(output_name=str(outputs[0]), expr=expr, inferred_shape=inferred_shape)}"
            )
        return True
    return False

def _emit_native_binary_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_index: int,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    runtime_shape_uncertain_tensors: Set[str],
    tensor_dtype_name_fn: Callable[[str], Optional[str]],
    binary_operand_expr_fn: Callable[[str, str], str],
    scalar_literal_expr_fn: Callable[[str], Optional[str]],
    can_emit_channel_first_binary_op_fn: Callable[[OperatorIR], bool],
    channel_first_binary_input_expr_fn: Callable[[str, str], Optional[str]],
    derived_local_var_name_fn: Callable[[str, str], str],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    target_shape_literal_fn: Callable[[str], str],
    emit_maybe_aligned_expr_fn: Callable[..., str],
    binary_runtime_shape_passthrough_operand_fn: Callable[[str, str], Optional[str]],
    binary_requires_runtime_alignment_fn: Callable[[str, str, str], bool],
    preferred_binary_alignment_anchor_fn: Callable[[str, str, str], Optional[str]],
    activation_lines_fn: Callable[[str, str], List[str]],
) -> bool:
    op_type = str(op.op_type)
    if op_type not in _DIRECT_CODEGEN_BINARY_FUNCTIONS:
        return False
    fn_name = _DIRECT_CODEGEN_BINARY_FUNCTIONS[op_type]
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    lhs_name = str(op.inputs[0])
    rhs_name = str(op.inputs[1])
    lhs_dtype_name = tensor_dtype_name_fn(lhs_name)
    rhs_dtype_name = tensor_dtype_name_fn(rhs_name)
    lhs_expr = binary_operand_expr_fn(lhs_name, rhs_name)
    rhs_scalar_literal = scalar_literal_expr_fn(rhs_name)
    rhs_scalar_literal_value = rhs_scalar_literal
    if op_type in {"MAXIMUM", "MINIMUM"}:
        rhs_scalar_literal = None
    rhs_expr = rhs_scalar_literal or binary_operand_expr_fn(rhs_name, lhs_name)
    if (
        rhs_scalar_literal_value is not None
        and op_type in {"MAXIMUM", "MINIMUM"}
    ):
        rhs_expr = (
            f"torch.as_tensor({rhs_scalar_literal_value}, "
            f"dtype={lhs_expr}.dtype, device={lhs_expr}.device)"
        )
    rhs_tensor = model_ir.tensors.get(rhs_name, None)
    if (
        op_type in {"MAXIMUM", "MINIMUM"}
        and rhs_tensor is not None
        and isinstance(rhs_tensor.data, np.ndarray)
        and int(np.size(rhs_tensor.data)) == 1
    ):
        rhs_literal = _python_literal_for_constant_tensor(rhs_tensor)
        if rhs_literal is not None:
            rhs_expr = (
                f"torch.as_tensor({rhs_literal}, "
                f"dtype={lhs_expr}.dtype, device={lhs_expr}.device)"
            )
    if (
        rhs_scalar_literal in {"True", "False"}
        and op_type in {"EQUAL", "NOT_EQUAL"}
    ):
        rhs_expr = (
            f"torch.as_tensor({rhs_scalar_literal}, "
            f"dtype={lhs_expr}.dtype, device={lhs_expr}.device)"
        )
    is_integer_div = (
        op_type == "DIV"
        and lhs_dtype_name in {"INT8", "INT16", "INT32", "INT64", "UINT8"}
        and rhs_dtype_name in {"INT8", "INT16", "INT32", "INT64", "UINT8"}
    )
    runtime_shape_passthrough_operand = binary_runtime_shape_passthrough_operand_fn(lhs_name, rhs_name)
    requires_runtime_alignment = binary_requires_runtime_alignment_fn(
        lhs_name, rhs_name, str(outputs[0])
    )

    def _binary_expr(left_expr: str, right_expr: str) -> str:
        if is_integer_div:
            return (
                f"torch.div({left_expr}, {right_expr}, "
                "rounding_mode='trunc')"
            )
        return f"{fn_name}({left_expr}, {right_expr})"

    output_name = str(outputs[0])
    output_tensor = model_ir.tensors.get(output_name, None)
    output_rank = len(list(output_tensor.shape)) if output_tensor is not None else 0
    output_layout = normalize_logical_layout(
        output_tensor.logical_layout if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
    )
    lhs_cf_expr: Optional[str] = None
    rhs_cf_expr: Optional[str] = None
    can_emit_channel_first_binary = (
        runtime_shape_passthrough_operand is None
        and not requires_runtime_alignment
        and can_emit_channel_first_binary_op_fn(op)
    )
    if (
        not can_emit_channel_first_binary
        and runtime_shape_passthrough_operand is None
        and output_tensor is not None
        and output_rank in {3, 4, 5}
        and (
            is_channel_last_logical_layout(output_layout)
            or is_channel_first_logical_layout(output_layout)
            or output_layout == LOGICAL_LAYOUT_UNKNOWN
        )
    ):
        lhs_cf_expr = channel_first_binary_input_expr_fn(lhs_name, rhs_name)
        rhs_cf_expr = channel_first_binary_input_expr_fn(rhs_name, lhs_name)
        can_emit_channel_first_binary = lhs_cf_expr is not None and rhs_cf_expr is not None
    if rhs_scalar_literal_value is not None and op_type in {"MAXIMUM", "MINIMUM"}:
        can_emit_channel_first_binary = False

    if can_emit_channel_first_binary:
        output_var = output_vars[0]
        raw_output_var = (
            output_var
            if output_layout == LOGICAL_LAYOUT_UNKNOWN or output_name in set(model_ir.outputs)
            else derived_local_var_name_fn(f"{output_var}_cf", "t")
        )
        if lhs_cf_expr is None:
            lhs_cf_expr = channel_first_binary_input_expr_fn(lhs_name, rhs_name)
        if rhs_cf_expr is None:
            rhs_cf_expr = channel_first_binary_input_expr_fn(rhs_name, lhs_name)
        if lhs_cf_expr is None or rhs_cf_expr is None:
            raise ModelIRPyTorchExportError(
                "Native PyTorch-like model.py codegen expected channel-first-capable binary inputs. "
                f"op={op_type} lhs={lhs_name} rhs={rhs_name}"
            )
        channel_first_tensor_expr_aliases[output_name] = raw_output_var
        forward_lines.append(f"{raw_output_var} = {_binary_expr(lhs_cf_expr, rhs_cf_expr)}")
        forward_lines.extend(activation_lines_fn(raw_output_var, fused))
        if (
            output_tensor is not None
            and is_channel_last_logical_layout(output_layout)
            and output_rank in {3, 4, 5}
        ):
            if can_omit_materialized_channel_last_alias_fn(output_name):
                return True
            perm_to_output = logical_layout_permutation(
                source_layout=channel_first_logical_layout(output_rank),
                target_layout=output_layout,
            )
            if perm_to_output is not None:
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_var} = _align_tensor_to_target_shape("
                    f"{raw_output_var}.permute({', '.join(str(int(v)) for v in perm_to_output)}).contiguous(), "
                    f"{target_shape_literal_fn(output_name)})"
                )
                return True
        return True
    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
    if rhs_scalar_literal is not None:
        forward_lines.append(
            f"{output_vars[0]} = {emit_maybe_aligned_expr_fn(output_name=outputs[0], expr=_binary_expr(lhs_expr, rhs_expr), inferred_shape=None)}"
        )
    elif runtime_shape_passthrough_operand is not None:
        forward_lines.append(
            f"{output_vars[0]} = {_binary_expr(lhs_expr, rhs_expr)}"
        )
    elif requires_runtime_alignment:
        lhs_uncertain = lhs_name in runtime_shape_uncertain_tensors
        rhs_uncertain = rhs_name in runtime_shape_uncertain_tensors
        preferred_anchor = preferred_binary_alignment_anchor_fn(lhs_name, rhs_name, str(outputs[0]))
        lhs_var = f"_binary_lhs_{op_index}"
        rhs_var = f"_binary_rhs_{op_index}"
        if lhs_uncertain ^ rhs_uncertain or preferred_anchor is not None:
            runtime_imports.add("_align_binary_inputs_to_anchor")
            if lhs_uncertain or preferred_anchor == "lhs":
                forward_lines.append(
                    f"{lhs_var}, {rhs_var} = _align_binary_inputs_to_anchor({lhs_expr}, {binary_operand_expr_fn(rhs_name, lhs_name)}, {output_target_shape})"
                )
            else:
                forward_lines.append(
                    f"{rhs_var}, {lhs_var} = _align_binary_inputs_to_anchor({binary_operand_expr_fn(rhs_name, lhs_name)}, {lhs_expr}, {output_target_shape})"
                )
        else:
            runtime_imports.add("_align_binary_inputs")
            forward_lines.append(
                f"{lhs_var}, {rhs_var} = _align_binary_inputs({lhs_expr}, {binary_operand_expr_fn(rhs_name, lhs_name)}, {output_target_shape})"
            )
        forward_lines.append(
            f"{output_vars[0]} = {emit_maybe_aligned_expr_fn(output_name=outputs[0], expr=_binary_expr(lhs_var, rhs_var), inferred_shape=None)}"
        )
    else:
        forward_lines.append(
            f"{output_vars[0]} = {emit_maybe_aligned_expr_fn(output_name=outputs[0], expr=_binary_expr(lhs_expr, rhs_expr), inferred_shape=None)}"
        )
    forward_lines.extend(activation_lines_fn(output_vars[0], fused))
    return True

def _emit_native_unary_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    channel_first_passthrough_input_expr_fn: Callable[[str], Optional[str]],
    can_emit_channel_first_shape_preserving_unary_op_fn: Callable[[OperatorIR], bool],
    derived_local_var_name_fn: Callable[[str, str], str],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    target_shape_literal_fn: Callable[[str], str],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    should_skip_align_for_shape_preserving_unary_fn: Callable[[str, str], bool],
    emit_maybe_aligned_expr_fn: Callable[..., str],
) -> bool:
    op_type = str(op.op_type)
    if op_type not in _DIRECT_CODEGEN_UNARY_EXPRESSIONS:
        return False
    template = _DIRECT_CODEGEN_UNARY_EXPRESSIONS[op_type]
    input_name = str(op.inputs[0])
    output_name = str(outputs[0])
    channel_first_input_expr = channel_first_passthrough_input_expr_fn(input_name)
    output_tensor = model_ir.tensors.get(output_name, None)
    output_layout = (
        normalize_logical_layout(output_tensor.logical_layout)
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    if (
        can_emit_channel_first_shape_preserving_unary_op_fn(op)
        and channel_first_input_expr is not None
        and output_tensor is not None
    ):
        if op_type == "LEAKY_RELU":
            channel_first_expr = template.format(
                x=channel_first_input_expr,
                alpha=float(op.options.get("alpha", 0.2)),
            )
        else:
            channel_first_expr = template.format(x=channel_first_input_expr)
        output_rank = len(list(output_tensor.shape))
        if is_channel_first_logical_layout(output_layout):
            channel_first_tensor_expr_aliases.pop(output_name, None)
            forward_lines.append(f"{output_vars[0]} = {channel_first_expr}")
            return True
        raw_output_var = (
            output_vars[0]
            if output_layout == LOGICAL_LAYOUT_UNKNOWN
            else derived_local_var_name_fn(f"{output_vars[0]}_cf", "t")
        )
        channel_first_tensor_expr_aliases[output_name] = raw_output_var
        forward_lines.append(f"{raw_output_var} = {channel_first_expr}")
        if raw_output_var != output_vars[0]:
            if can_omit_materialized_channel_last_alias_fn(output_name):
                return True
            perm_to_output = logical_layout_permutation(
                source_layout=channel_first_logical_layout(output_rank),
                target_layout=output_layout,
            )
            if perm_to_output is None:
                raise ModelIRPyTorchExportError(
                    "Native PyTorch-like model.py codegen could not derive a unary layout bridge. "
                    f"output={output_name} output_layout={output_layout} rank={output_rank}"
                )
            runtime_imports.add("_align_tensor_to_target_shape")
            forward_lines.append(
                f"{output_vars[0]} = _align_tensor_to_target_shape("
                f"{raw_output_var}.permute({', '.join(str(int(v)) for v in perm_to_output)}).contiguous(), "
                f"{target_shape_literal_fn(output_name)})"
            )
        return True
    if op_type == "LEAKY_RELU":
        expr = template.format(x=tensor_expr_fn(str(op.inputs[0])), alpha=float(op.options.get("alpha", 0.2)))
    else:
        expr = template.format(x=tensor_expr_fn(str(op.inputs[0])))
    channel_first_tensor_expr_aliases.pop(output_name, None)
    inferred_shape = tensor_shape_list_fn(str(op.inputs[0]))
    if should_skip_align_for_shape_preserving_unary_fn(str(op.inputs[0]), str(outputs[0])):
        forward_lines.append(f"{output_vars[0]} = {expr}")
    else:
        forward_lines.append(
            f"{output_vars[0]} = {emit_maybe_aligned_expr_fn(output_name=outputs[0], expr=expr, inferred_shape=inferred_shape)}"
        )
    return True

def _emit_native_transpose_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    preserve_channel_last_tensor_names: Set[str],
    consumer_index: Dict[str, List[int]],
    producer_index: Dict[str, int],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    tensor_expr_for_channel_first_bridge_fn: Callable[[str, Sequence[int]], Optional[str]],
    can_fold_channel_last_alias_slice_consumer_fn: Callable[..., bool],
    all_consumers_are_channel_first_binary_ops_fn: Callable[[str], bool],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    has_channel_last_consumer_hint_for_same_shape_transpose_fn: Callable[[OperatorIR], bool],
    is_batchless_rank3_public_output_transpose_fn: Callable[[OperatorIR], bool],
    target_shape_literal_fn: Callable[[str], str],
) -> bool:
    if str(op.op_type) != "TRANSPOSE":
        return False
    transpose_related_names = {str(v) for v in list(op.inputs) + list(op.outputs)}
    stale_channel_last_transpose = has_channel_last_consumer_hint_for_same_shape_transpose_fn(op)
    batchless_public_output_transpose = is_batchless_rank3_public_output_transpose_fn(op)
    transpose_perm = _read_transpose_perm(model_ir, op)
    transpose_input_name = str(op.inputs[0]) if len(op.inputs) >= 1 else ""
    transpose_output_name = str(outputs[0]) if len(outputs) == 1 else ""
    transpose_input_tensor = model_ir.tensors.get(transpose_input_name, None)
    transpose_output_tensor = model_ir.tensors.get(transpose_output_name, None)
    transpose_consumer_indices = [int(v) for v in consumer_index.get(transpose_output_name, [])]
    reshape_only_consumers = (
        len(transpose_consumer_indices) > 0
        and all(
            str(model_ir.operators[int(consumer_idx)].op_type) == "RESHAPE"
            for consumer_idx in transpose_consumer_indices
        )
    )
    inconsistent_same_layout_transpose = _is_inconsistent_same_layout_transpose(
        input_tensor=model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None,
        output_tensor=model_ir.tensors.get(str(outputs[0]), None) if len(outputs) == 1 else None,
        perm=transpose_perm,
    )
    allow_transpose_elision = stale_channel_last_transpose or not any(
        name in preserve_channel_last_tensor_names for name in transpose_related_names
    )
    if batchless_public_output_transpose or allow_transpose_elision and (
        stale_channel_last_transpose or
        _is_reshape_only_residual_layout_bridge_transpose(
            model_ir=model_ir,
            op=op,
            consumers=consumer_index,
        ) or _is_inconsistent_standard_layout_transpose(
            input_tensor=model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None,
            output_tensor=model_ir.tensors.get(str(outputs[0]), None) if len(outputs) == 1 else None,
            perm=transpose_perm,
        ) or _is_inconsistent_same_layout_transpose(
            input_tensor=model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None,
            output_tensor=model_ir.tensors.get(str(outputs[0]), None) if len(outputs) == 1 else None,
            perm=transpose_perm,
        ) and not reshape_only_consumers
    ):
        forward_lines.append(
            f"{output_vars[0]} = {tensor_expr_fn(str(op.inputs[0]))}"
        )
        return True
    folded_channel_first_expr = (
        None
        if transpose_perm is None
        else tensor_expr_for_channel_first_bridge_fn(
            transpose_input_name,
            transpose_perm,
        )
    )
    if folded_channel_first_expr is not None:
        output_layout = normalize_logical_layout(
            transpose_output_tensor.logical_layout if transpose_output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        )
        if transpose_output_tensor is not None and output_layout == LOGICAL_LAYOUT_UNKNOWN:
            channel_first_tensor_expr_aliases[transpose_output_name] = output_vars[0]
        else:
            channel_first_tensor_expr_aliases.pop(transpose_output_name, None)
        forward_lines.append(f"{output_vars[0]} = {folded_channel_first_expr}")
        return True
    if transpose_input_tensor is not None and transpose_output_tensor is not None:
        rank = len(list(transpose_input_tensor.shape))
        expected_cf_to_cl_perm = _perm_cf_to_cl(rank)
        input_cf_expr = channel_first_tensor_expr_aliases.get(transpose_input_name, None)
        if input_cf_expr is None and is_channel_first_logical_layout(
            normalize_logical_layout(transpose_input_tensor.logical_layout)
        ):
            input_cf_expr = tensor_expr_fn(transpose_input_name)
        if (
            input_cf_expr is not None
            and expected_cf_to_cl_perm is not None
            and list(transpose_perm or []) == list(expected_cf_to_cl_perm)
            and len(consumer_index.get(transpose_output_name, [])) > 0
            and all(
                can_fold_channel_last_alias_slice_consumer_fn(
                    model_ir.operators[int(consumer_idx)],
                    expected_input_name=transpose_output_name,
                )
                for consumer_idx in consumer_index.get(transpose_output_name, [])
            )
        ):
            channel_first_tensor_expr_aliases[transpose_output_name] = str(input_cf_expr)
            return True
        if (
            input_cf_expr is not None
            and expected_cf_to_cl_perm is not None
            and list(transpose_perm or []) == list(expected_cf_to_cl_perm)
            and all_consumers_are_channel_first_binary_ops_fn(transpose_output_name)
        ):
            channel_first_tensor_expr_aliases[transpose_output_name] = str(input_cf_expr)
            return True
        if (
            input_cf_expr is not None
            and expected_cf_to_cl_perm is not None
            and list(transpose_perm or []) == list(expected_cf_to_cl_perm)
            and can_omit_materialized_channel_last_alias_fn(transpose_output_name)
        ):
            channel_first_tensor_expr_aliases[transpose_output_name] = str(input_cf_expr)
            return True
    runtime_imports.add("_shape_list")
    runtime_imports.add("_torch_permute")
    if len(op.inputs) >= 2:
        const_perm_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
        if const_perm_values is not None:
            perm_values = [int(v) for v in list(const_perm_values)]
            input_layout = normalize_logical_layout(
                transpose_input_tensor.logical_layout
                if transpose_input_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            output_layout = normalize_logical_layout(
                transpose_output_tensor.logical_layout
                if transpose_output_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            if (
                reshape_only_consumers
                and input_layout != LOGICAL_LAYOUT_UNKNOWN
                and input_layout == output_layout
            ):
                forward_lines.append(
                    f"{output_vars[0]} = {tensor_expr_fn(str(op.inputs[0]))}.permute({', '.join(str(int(v)) for v in perm_values)}).contiguous()"
                )
                return True
            perm_expr = repr(perm_values)
        else:
            perm_expr = f"_shape_list({tensor_expr_fn(str(op.inputs[1]))})"
    else:
        perm_expr = repr([int(v) for v in list(op.options.get('perm', []))])
    forward_lines.append(
        f"{output_vars[0]} = _torch_permute({tensor_expr_fn(str(op.inputs[0]))}, {perm_expr})"
    )
    return True

def _emit_native_shape_transform_misc_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    axis_expr_from_input_fn: Callable[..., str],
) -> bool:
    if op_type == "REVERSE_V2":
        data_expr = tensor_expr_fn(str(op.inputs[0]))
        axes_expr = tensor_expr_fn(str(op.inputs[1]))
        axes_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
        data_rank = len(list(model_ir.tensors[str(op.inputs[0])].shape))
        if axes_values is not None:
            normalized_dims = [
                int(axis) if int(axis) >= 0 else int(axis) + int(data_rank)
                for axis in list(axes_values)
            ]
            dims_expr = repr(normalized_dims)
        else:
            dims_expr = (
                f"[int(v) if int(v) >= 0 else int(v) + {data_expr}.ndim "
                f"for v in {axes_expr}.to(dtype=torch.int64).reshape(-1)]"
            )
        forward_lines.append(
            f"{output_vars[0]} = torch.flip("
            f"{data_expr}, "
            f"dims={dims_expr}"
            f")"
        )
        return True
    if op_type == "EXPAND_DIMS":
        axis_expr = (
            axis_expr_from_input_fn(str(op.inputs[1]), device_expr=tensor_expr_fn(str(op.inputs[0])))
            if len(op.inputs) >= 2
            else repr(int(op.options.get("axis", 0)))
        )
        forward_lines.append(
            f"{output_vars[0]} = torch.unsqueeze({tensor_expr_fn(str(op.inputs[0]))}, dim={axis_expr})"
        )
        return True
    if op_type == "SQUEEZE":
        squeeze_dims = [int(v) for v in list(op.options.get("squeezeDims", []))]
        if len(squeeze_dims) == 0:
            forward_lines.append(f"{output_vars[0]} = torch.squeeze({tensor_expr_fn(str(op.inputs[0]))})")
        else:
            runtime_imports.add("_normalize_dim")
            forward_lines.append(f"{output_vars[0]} = {tensor_expr_fn(str(op.inputs[0]))}")
            for axis in sorted(squeeze_dims, reverse=True):
                forward_lines.append(
                    f"{output_vars[0]} = torch.squeeze({output_vars[0]}, dim=_normalize_dim({int(axis)}, {output_vars[0]}.ndim))"
                )
        return True
    if op_type == "PACK":
        axis = int(op.options.get("axis", 0))
        inputs_expr = ", ".join(tensor_expr_fn(str(name)) for name in op.inputs)
        forward_lines.append(
            f"{output_vars[0]} = torch.stack([{inputs_expr}], dim={axis})"
        )
        return True
    if op_type == "UNPACK":
        runtime_imports.add("_normalize_dim")
        axis = int(op.options.get("axis", 0))
        input_expr = tensor_expr_fn(str(op.inputs[0]))
        forward_lines.append(
            f"{', '.join(output_vars)} = list(torch.unbind({input_expr}, dim=_normalize_dim({axis}, {input_expr}.ndim)))"
        )
        return True
    if op_type == "SPLIT":
        runtime_imports.add("_normalize_dim")
        data_expr = tensor_expr_fn(str(op.inputs[-1]))
        if len(op.inputs) >= 2:
            axis_expr = axis_expr_from_input_fn(str(op.inputs[0]), device_expr=data_expr)
        else:
            axis_expr = repr(int(op.options.get("axis", 0)))
        sections = int(op.options.get("numSplits", len(outputs)))
        forward_lines.append(
            f"{', '.join(output_vars)} = list(torch.tensor_split({data_expr}, {sections}, dim=_normalize_dim({axis_expr}, {data_expr}.ndim)))"
        )
        return True
    return False

def _emit_native_concat_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_index: int,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    derived_local_var_name_fn: Callable[[str, str], str],
    activation_lines_fn: Callable[[str, str], List[str]],
    resolve_concat_axis_for_channel_first_fn: Callable[[OperatorIR], Optional[Tuple[int, List[int], List[int]]]],
    channel_first_concat_input_expr_fn: Callable[[str], Optional[str]],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    target_shape_literal_fn: Callable[[str], str],
    tensor_exact_static_shape_list_fn: Callable[[str], Optional[List[int]]],
    target_shape_values_fn: Callable[[str], Optional[List[int]]],
) -> bool:
    if str(op.op_type) != "CONCATENATION":
        return False
    axis = int(op.options.get("axis", 0))
    gather_elements_axis_coord_input_index = next(
        (idx for idx, name in enumerate(op.inputs) if str(name).endswith("_gather_elements_axis_coord")),
        None,
    )
    gather_elements_coords_concat = bool(
        str(outputs[0]).endswith("_gather_elements_coords")
        and gather_elements_axis_coord_input_index is not None
        and axis == len(op.inputs)
    )
    if gather_elements_coords_concat:
        assert gather_elements_axis_coord_input_index is not None
        axis_coord_expr = tensor_expr_fn(str(op.inputs[gather_elements_axis_coord_input_index]))
        coord_shape_var = f"_gather_elements_coords_shape_{op_index}"
        coord_vars: List[str] = []
        forward_lines.append(
            f"{coord_shape_var} = [int(v) for v in list({axis_coord_expr}.shape[:-1])]"
        )
        for dim_index in range(len(op.inputs)):
            if dim_index == gather_elements_axis_coord_input_index:
                coord_vars.append(axis_coord_expr)
                continue
            coord_var = f"_gather_elements_coord_{op_index}_{dim_index}"
            view_shape_var = f"_gather_elements_coord_view_shape_{op_index}_{dim_index}"
            forward_lines.append(f"{view_shape_var} = [1] * {len(op.inputs)}")
            forward_lines.append(
                f"{view_shape_var}[{dim_index}] = {coord_shape_var}[{dim_index}]"
            )
            forward_lines.append(
                f"{coord_var} = torch.arange({coord_shape_var}[{dim_index}], dtype={axis_coord_expr}.dtype, device={axis_coord_expr}.device).reshape(*{view_shape_var}, 1).expand(*{coord_shape_var}, 1)"
            )
            coord_vars.append(coord_var)
        forward_lines.append(
            f"{output_vars[0]} = torch.cat([{', '.join(coord_vars)}], dim={len(op.inputs)})"
        )
        return True
    concat_cf_spec = resolve_concat_axis_for_channel_first_fn(op)
    concat_cf_inputs = [
        channel_first_concat_input_expr_fn(str(name))
        for name in op.inputs
    ]
    output_tensor = model_ir.tensors.get(outputs[0], None)
    output_layout = (
        normalize_logical_layout(output_tensor.logical_layout)
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    output_rank = len(list(output_tensor.shape)) if output_tensor is not None else 0
    if (
        concat_cf_spec is not None
        and output_rank in {3, 4, 5}
        and output_layout in {
            LOGICAL_LAYOUT_UNKNOWN,
            channel_first_logical_layout(output_rank),
            channel_last_logical_layout(output_rank),
        }
        and all(input_expr is not None for input_expr in concat_cf_inputs)
    ):
        concat_cf_axis, concat_cf_output_shape, concat_perm_from_cf = concat_cf_spec
        fused = str(op.options.get("fusedActivationFunction", "NONE"))
        stored_output_shape = tensor_shape_list_fn(outputs[0]) or []
        raw_matches_stored_shape = (
            [int(v) for v in list(concat_cf_output_shape)]
            == [int(v) for v in list(stored_output_shape)]
        )
        needs_materialized_output_bridge = (
            len(concat_perm_from_cf) == output_rank
            and [int(v) for v in list(concat_perm_from_cf)] != [int(v) for v in list(range(output_rank))]
            and not raw_matches_stored_shape
        )
        if is_channel_first_logical_layout(output_layout) and raw_matches_stored_shape:
            raw_output_var = output_vars[0]
            channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
        elif output_layout == LOGICAL_LAYOUT_UNKNOWN and not needs_materialized_output_bridge:
            raw_output_var = output_vars[0]
            channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
        else:
            raw_output_var = derived_local_var_name_fn(f"{output_vars[0]}_cf", "t")
            channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
        forward_lines.append(
            f"{raw_output_var} = torch.cat([{', '.join(str(v) for v in concat_cf_inputs)}], dim={int(concat_cf_axis)})"
        )
        forward_lines.extend(activation_lines_fn(raw_output_var, fused))
        if raw_output_var != output_vars[0]:
            if can_omit_materialized_channel_last_alias_fn(outputs[0]):
                return True
            if needs_materialized_output_bridge:
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_vars[0]} = _align_tensor_to_target_shape("
                    f"{raw_output_var}.permute({', '.join(str(int(v)) for v in concat_perm_from_cf)}).contiguous(), "
                    f"{target_shape_literal_fn(outputs[0])})"
                )
            else:
                perm_to_output = logical_layout_permutation(
                    source_layout=channel_first_logical_layout(output_rank),
                    target_layout=output_layout,
                )
                if perm_to_output is None:
                    raise ModelIRPyTorchExportError(
                        "Native PyTorch-like model.py codegen could not derive a channel-first concat bridge. "
                        f"output={outputs[0]} output_layout={output_layout} rank={output_rank}"
                    )
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_vars[0]} = _align_tensor_to_target_shape("
                    f"{raw_output_var}.permute({', '.join(str(int(v)) for v in perm_to_output)}).contiguous(), "
                    f"{target_shape_literal_fn(outputs[0])})"
                )
        return True
    inputs_expr = ", ".join(tensor_expr_fn(str(name)) for name in op.inputs)
    runtime_imports.add("_apply_concat")
    concat_expr = (
        f"_apply_concat([{inputs_expr}], axis={axis}, target_shape={output_target_shape}, "
        f"fused={str(op.options.get('fusedActivationFunction', 'NONE'))!r})"
    )
    exact_output_shape = tensor_exact_static_shape_list_fn(outputs[0])
    target_output_shape = target_shape_values_fn(outputs[0])
    if (
        exact_output_shape is not None
        and (
            target_output_shape is None
            or [int(v) for v in list(exact_output_shape)] != [int(v) for v in list(target_output_shape)]
        )
    ):
        concat_expr = f"torch.reshape({concat_expr}, {repr(exact_output_shape)})"
    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
    forward_lines.append(f"{output_vars[0]} = {concat_expr}")
    return True

def _shape_tensor_length_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[int]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    shape_values = [int(v) for v in list(tensor.shape)]
    if len(shape_values) == 0:
        return 0
    if len(shape_values) != 1 or int(shape_values[0]) < 0:
        return None
    return int(shape_values[0])

def _reconstruct_shape_scalar_expr_for_codegen(
    *,
    model_ir: ModelIR,
    producer_by_output_name: Dict[str, OperatorIR],
    tensor_exact_static_shape_list_fn: Callable[[str], Optional[List[int]]],
    tensor_expr_fn: Callable[[str], str],
    runtime_imports: Set[str],
    tensor_name: str,
    seen: Optional[Set[str]] = None,
) -> Optional[str]:
    current_name = str(tensor_name)
    if seen is None:
        seen = set()
    if current_name in seen:
        return None
    next_seen = set(seen)
    next_seen.add(current_name)

    tensor = model_ir.tensors.get(current_name, None)
    constant_values = _constant_int_list(tensor)
    if constant_values is not None and len(constant_values) == 1:
        return repr(int(constant_values[0]))

    producer = producer_by_output_name.get(current_name, None)
    if producer is None:
        return None
    op_type = str(producer.op_type)
    inputs = [str(v) for v in list(producer.inputs)]

    def _shape_dim_expr_from_shape_input(shape_tensor_name: str, dim_index: int) -> Optional[str]:
        shape_producer = producer_by_output_name.get(str(shape_tensor_name), None)
        if shape_producer is None:
            return None
        if str(shape_producer.op_type) != "SHAPE" or len(list(shape_producer.inputs)) < 1:
            return None
        source_name = str(shape_producer.inputs[0])
        exact_input_shape = tensor_exact_static_shape_list_fn(source_name)
        if exact_input_shape is not None and int(dim_index) < len(exact_input_shape):
            return repr(int(exact_input_shape[int(dim_index)]))
        return f"{tensor_expr_fn(source_name)}.shape[{int(dim_index)}]"

    if op_type in {"CAST", "IDENTITY", "RESHAPE", "SQUEEZE"} and len(inputs) >= 1:
        return _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
    if op_type == "SLICE" and len(inputs) >= 3:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        size_values = _constant_int_list(model_ir.tensors.get(inputs[2], None))
        output_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=current_name)
        if base_expr is not None and begin_values is not None and len(begin_values) == 1 and size_values is not None and len(size_values) == 1 and output_len == 1:
            shape_dim_expr = _shape_dim_expr_from_shape_input(inputs[0], int(begin_values[0]))
            if shape_dim_expr is not None:
                return shape_dim_expr
            return f"({base_expr})[{int(begin_values[0])}]"
    if op_type == "STRIDED_SLICE" and len(inputs) >= 4:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        stride_values = _constant_int_list(model_ir.tensors.get(inputs[3], None))
        output_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=current_name)
        if base_expr is not None and begin_values is not None and len(begin_values) == 1 and stride_values is not None and len(stride_values) == 1 and output_len == 1:
            shape_dim_expr = _shape_dim_expr_from_shape_input(inputs[0], int(begin_values[0]))
            if shape_dim_expr is not None:
                return shape_dim_expr
            return f"({base_expr})[{int(begin_values[0])}]"
    if op_type == "GATHER" and len(inputs) >= 2 and int(producer.options.get("axis", 0)) == 0:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        index_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        if base_expr is not None and index_values is not None and len(index_values) == 1:
            shape_dim_expr = _shape_dim_expr_from_shape_input(inputs[0], int(index_values[0]))
            if shape_dim_expr is not None:
                return shape_dim_expr
            return f"({base_expr})[{int(index_values[0])}]"
    if op_type == "EQUAL" and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            return f"({lhs_expr} == {rhs_expr})"
    if op_type == "NOT_EQUAL" and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            return f"({lhs_expr} != {rhs_expr})"
    if op_type in {"MAXIMUM", "MINIMUM"} and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            fn_name = "max" if op_type == "MAXIMUM" else "min"
            return f"{fn_name}({lhs_expr}, {rhs_expr})"
    if op_type in {"ADD", "SUB", "MUL"} and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            op_symbol = "+" if op_type == "ADD" else "-" if op_type == "SUB" else "*"
            return f"({lhs_expr} {op_symbol} {rhs_expr})"
    if op_type == "SELECT" and len(inputs) >= 3:
        return None
    if op_type == "REDUCE_PROD" and len(inputs) >= 1:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        axes_values = _constant_int_list(model_ir.tensors.get(inputs[1], None)) if len(inputs) >= 2 else None
        output_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=current_name)
        input_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=inputs[0])
        if (
            base_expr is not None
            and axes_values == [0]
            and output_len == 1
            and input_len is not None
            and input_len >= 0
        ):
            return _product_expr([f"({base_expr})[{index}]" for index in range(int(input_len))])
    return None

def _reconstruct_shape_list_expr_for_codegen(
    *,
    model_ir: ModelIR,
    producer_by_output_name: Dict[str, OperatorIR],
    tensor_exact_static_shape_list_fn: Callable[[str], Optional[List[int]]],
    tensor_expr_fn: Callable[[str], str],
    runtime_imports: Set[str],
    tensor_name: str,
    seen: Optional[Set[str]] = None,
) -> Optional[str]:
    current_name = str(tensor_name)
    if seen is None:
        seen = set()
    if current_name in seen:
        return None
    next_seen = set(seen)
    next_seen.add(current_name)

    tensor = model_ir.tensors.get(current_name, None)
    constant_values = _constant_int_list(tensor)
    if constant_values is not None:
        return repr([int(v) for v in list(constant_values)])

    producer = producer_by_output_name.get(current_name, None)
    if producer is None:
        return None
    op_type = str(producer.op_type)
    inputs = [str(v) for v in list(producer.inputs)]
    if op_type == "CAST" and len(inputs) >= 1:
        input_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        if input_expr is not None:
            return input_expr
        input_scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        if input_scalar_expr is not None:
            return f"[{input_scalar_expr}]"
        return None
    if op_type == "IDENTITY" and len(inputs) >= 1:
        return _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
    if op_type == "RESHAPE" and len(inputs) >= 2:
        target_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if target_expr is not None:
            return target_expr
        target_scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if target_scalar_expr is not None:
            return f"[{target_scalar_expr}]"
        return None
    if op_type == "SHAPE" and len(inputs) >= 1:
        exact_input_shape = tensor_exact_static_shape_list_fn(inputs[0])
        if exact_input_shape is not None:
            return repr([int(v) for v in list(exact_input_shape)])
        tensor_expr = tensor_expr_fn(inputs[0])
        runtime_imports.add("_tensor_shape_list")
        return f"_tensor_shape_list({tensor_expr})"
    if op_type == "SLICE" and len(inputs) >= 3:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        size_values = _constant_int_list(model_ir.tensors.get(inputs[2], None))
        if base_expr is not None and begin_values is not None and len(begin_values) == 1 and size_values is not None and len(size_values) == 1:
            start = int(begin_values[0])
            length = int(size_values[0])
            stop_expr = "" if length < 0 else str(start + length)
            return f"({base_expr})[{start}:{stop_expr}]"
    if op_type == "STRIDED_SLICE" and len(inputs) >= 4:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        end_values = _constant_int_list(model_ir.tensors.get(inputs[2], None))
        stride_values = _constant_int_list(model_ir.tensors.get(inputs[3], None))
        if (
            base_expr is not None
            and begin_values is not None and len(begin_values) == 1
            and end_values is not None and len(end_values) == 1
            and stride_values is not None and len(stride_values) == 1
        ):
            start = int(begin_values[0])
            end_value = int(end_values[0])
            step = int(stride_values[0])
            stop_expr = "" if bool(producer.options.get("endMask", 0)) else str(end_value)
            if step == 1:
                return f"({base_expr})[{start}:{stop_expr}]"
            return f"({base_expr})[{start}:{stop_expr}:{step}]"
    if op_type == "GATHER" and len(inputs) >= 2 and int(producer.options.get("axis", 0)) == 0:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        index_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        if base_expr is not None and index_values is not None:
            parts = ", ".join(f"({base_expr})[{int(index)}]" for index in index_values)
            return f"[{parts}]"
    if op_type == "CONCATENATION":
        part_exprs: List[str] = []
        for input_name in inputs:
            input_expr = _reconstruct_shape_list_expr_for_codegen(
                model_ir=model_ir,
                producer_by_output_name=producer_by_output_name,
                tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
                tensor_expr_fn=tensor_expr_fn,
                runtime_imports=runtime_imports,
                tensor_name=input_name,
                seen=next_seen,
            )
            if input_expr is None:
                input_scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
                    model_ir=model_ir,
                    producer_by_output_name=producer_by_output_name,
                    tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
                    tensor_expr_fn=tensor_expr_fn,
                    runtime_imports=runtime_imports,
                    tensor_name=input_name,
                    seen=next_seen,
                )
                if input_scalar_expr is None:
                    return None
                input_expr = f"[{input_scalar_expr}]"
            part_exprs.append(f"({input_expr})")
        if len(part_exprs) == 0:
            return "[]"
        combined_expr = part_exprs[0]
        for part_expr in part_exprs[1:]:
            combined_expr = f"({combined_expr} + {part_expr})"
        return combined_expr
    if op_type in {"SELECT", "REDUCE_PROD"}:
        scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=current_name,
            seen=next_seen,
        )
        if scalar_expr is not None:
            return f"[{scalar_expr}]"
    return None

def _has_channel_last_consumer_hint_for_same_shape_transpose_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    op: OperatorIR,
) -> bool:
    if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 1 or len(op.outputs) != 1:
        return False
    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    perm = _read_transpose_perm(model_ir, op)
    if input_tensor is None or output_tensor is None or perm is None:
        return False
    rank = len(list(output_tensor.shape))
    if rank not in {3, 4, 5}:
        return False
    input_layout = normalize_logical_layout(getattr(input_tensor, "logical_layout", None))
    output_layout = normalize_logical_layout(getattr(output_tensor, "logical_layout", None))
    layouts_disagree = (
        input_layout != LOGICAL_LAYOUT_UNKNOWN
        and output_layout != LOGICAL_LAYOUT_UNKNOWN
        and input_layout != output_layout
    )
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if input_shape != output_shape:
        return False
    if not _is_standard_channel_layout_permutation(perm=perm, rank=rank):
        return False
    reduction_spatial_axes = {
        3: [1],
        4: [1, 2],
        5: [1, 2, 3],
    }.get(rank, [])
    passthrough_ops = {
        "ABS",
        "CAST",
        "IDENTITY",
        "LEAKY_RELU",
        "LOGISTIC",
        "NEG",
        "RELU",
        "RELU6",
        "RESHAPE",
        "SIGMOID",
        "SQRT",
        "SQUARE",
        "TANH",
    }
    reduction_ops = {"MEAN", "SUM", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}
    broadcast_hint_ops = {"ADD", "DIV", "MAXIMUM", "MINIMUM", "MUL", "SUB"}
    worklist: List[str] = [str(op.outputs[0])]
    visited: Set[str] = set()
    while len(worklist) > 0:
        current_name = str(worklist.pop())
        if current_name in visited:
            continue
        visited.add(current_name)
        for consumer_idx in consumer_index.get(current_name, []):
            consumer = model_ir.operators[int(consumer_idx)]
            consumer_type = str(consumer.op_type)
            if consumer_type in passthrough_ops and len(consumer.outputs) == 1:
                worklist.append(str(consumer.outputs[0]))
                continue
            if consumer_type in reduction_ops and len(consumer.inputs) >= 2 and str(consumer.inputs[0]) == current_name:
                axes_values = _constant_int_list(model_ir.tensors.get(str(consumer.inputs[1]), None))
                if axes_values == reduction_spatial_axes:
                    return True
                continue
            if consumer_type in broadcast_hint_ops and current_name in {str(v) for v in consumer.inputs}:
                if layouts_disagree:
                    continue
                for other_name in consumer.inputs:
                    other_name = str(other_name)
                    if other_name == current_name:
                        continue
                    other_tensor = model_ir.tensors.get(other_name, None)
                    if other_tensor is None or not isinstance(other_tensor.data, np.ndarray):
                        continue
                    other_shape = [int(v) for v in list(other_tensor.shape)]
                    if len(other_shape) != rank:
                        continue
                    if all(int(v) == 1 for v in other_shape[:-1]) and int(other_shape[-1]) > 1:
                        return True
    return False

def _is_batchless_rank3_public_output_transpose_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    batchless_rank3_public_boundary_names: Set[str],
    op: OperatorIR,
) -> bool:
    if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 1 or len(op.outputs) != 1:
        return False
    output_name = str(op.outputs[0])
    if output_name not in {str(name) for name in list(model_ir.outputs)}:
        return False
    input_name = str(op.inputs[0])
    input_tensor = model_ir.tensors.get(input_name, None)
    output_tensor = model_ir.tensors.get(output_name, None)
    perm = _read_transpose_perm(model_ir, op)
    if input_tensor is None or output_tensor is None or perm != [0, 2, 1]:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if len(input_shape) != 3 or input_shape != output_shape or int(input_shape[0]) != 1:
        return False
    if (
        output_name in batchless_rank3_public_boundary_names
        or input_name in batchless_rank3_public_boundary_names
    ):
        return True

    current_names: List[str] = [input_name]
    visited_names: Set[str] = set()
    unary_passthrough_ops = {
        "ABS",
        "CAST",
        "IDENTITY",
        "LEAKY_RELU",
        "LOGISTIC",
        "NEG",
        "RELU",
        "RELU6",
        "TANH",
    }
    for _ in range(6):
        next_names: List[str] = []
        for current_name in current_names:
            if current_name in visited_names:
                continue
            visited_names.add(current_name)
            producer_idx = producer_index.get(current_name, None)
            if producer_idx is None:
                continue
            producer_op = model_ir.operators[int(producer_idx)]
            producer_type = str(producer_op.op_type)
            if producer_type == "RESHAPE" and len(producer_op.inputs) >= 1:
                source_tensor = model_ir.tensors.get(str(producer_op.inputs[0]), None)
                reshaped_tensor = model_ir.tensors.get(str(producer_op.outputs[0]), None)
                if source_tensor is not None and reshaped_tensor is not None:
                    source_shape = [int(v) for v in list(source_tensor.shape)]
                    reshaped_shape = [int(v) for v in list(reshaped_tensor.shape)]
                    if (
                        len(source_shape) == 4
                        and len(reshaped_shape) == 3
                        and int(source_shape[0]) == 1
                        and int(reshaped_shape[0]) == 1
                        and (
                            (
                                int(source_shape[1]) == 1
                                and int(source_shape[2]) == int(reshaped_shape[1])
                                and int(source_shape[3]) == int(reshaped_shape[2])
                            ) or (
                                int(source_shape[3]) == 1
                                and int(source_shape[1]) == int(reshaped_shape[1])
                                and int(source_shape[2]) == int(reshaped_shape[2])
                            )
                        )
                    ):
                        return True
                next_names.append(str(producer_op.inputs[0]))
                continue
            if producer_type in unary_passthrough_ops and len(producer_op.inputs) >= 1:
                next_names.append(str(producer_op.inputs[0]))
                continue
            if producer_type == "BATCH_MATMUL" and len(producer_op.inputs) >= 1:
                next_names.append(str(producer_op.inputs[0]))
                continue
            if producer_type == "ADD":
                dynamic_inputs = [
                    str(name)
                    for name in list(producer_op.inputs)
                    if (
                        (input_tensor := model_ir.tensors.get(str(name), None)) is None
                        or not isinstance(input_tensor.data, np.ndarray)
                    )
                ]
                if len(dynamic_inputs) == 1:
                    next_names.append(dynamic_inputs[0])
                    continue
        current_names = next_names
    return False

def _prune_dead_forward_lines(
    lines: Sequence[str],
    *,
    input_var_names: Sequence[str],
    output_var_names: Sequence[str],
) -> List[str]:
    if len(lines) == 0:
        return []

    parsed_statements: List[ast.stmt] = []
    top_level_assigned_names: List[List[str]] = []
    raw_used_names: List[List[str]] = []
    for line in lines:
        statement = ast.parse(str(line)).body[0]
        parsed_statements.append(statement)
        top_level_assigned_names.append(_extract_statement_assignments(statement))
        raw_used_names.append(_extract_statement_loads(statement))

    local_name_candidates: Set[str] = {str(name) for name in list(input_var_names)}
    local_name_candidates.update(str(name) for name in list(output_var_names))
    for assigned_names in top_level_assigned_names:
        local_name_candidates.update(str(name) for name in assigned_names)

    assigned_names_by_line: List[List[str]] = []
    used_names_by_line: List[List[str]] = []
    for assigned_names, used_names in zip(top_level_assigned_names, raw_used_names):
        assigned_filtered = [str(name) for name in assigned_names if str(name) in local_name_candidates]
        used_filtered = [str(name) for name in used_names if str(name) in local_name_candidates]
        assigned_names_by_line.append(assigned_filtered)
        used_names_by_line.append(used_filtered)

    live_names: Set[str] = {str(name) for name in list(output_var_names)}
    kept_lines_reversed: List[str] = []
    for line_index in range(len(lines) - 1, -1, -1):
        assigned_names = assigned_names_by_line[line_index]
        used_names = used_names_by_line[line_index]
        if len(assigned_names) > 0 and all(str(name) not in live_names for name in assigned_names):
            continue
        kept_lines_reversed.append(str(lines[line_index]))
        for name in assigned_names:
            live_names.discard(str(name))
        live_names.update(str(name) for name in used_names)
    kept_lines_reversed.reverse()
    return kept_lines_reversed

def _fold_single_use_static_reshape_chains(
    lines: Sequence[str],
    *,
    tensor_var_names: Dict[str, str],
    model_ir: ModelIR,
) -> List[str]:
    rewritten = [str(line) for line in lines]
    if len(rewritten) < 2:
        return rewritten
    tensor_name_by_var_name = {str(var_name): str(tensor_name) for tensor_name, var_name in tensor_var_names.items()}

    def _reshape_call(value: ast.AST) -> Optional[Tuple[ast.expr, ast.expr]]:
        if not isinstance(value, ast.Call):
            return None
        if len(value.keywords) != 0 or len(value.args) < 2:
            return None
        func = value.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "reshape"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            return None
        return cast(ast.expr, value.args[0]), cast(ast.expr, value.args[1])

    def _static_shape_from_expr(
        shape_expr: ast.AST,
        *,
        input_name: str,
        input_shape: Sequence[int],
    ) -> Optional[List[int]]:
        try:
            literal_value = ast.literal_eval(shape_expr)
        except Exception:
            literal_value = None
        if isinstance(literal_value, (list, tuple)):
            values = [int(v) for v in list(literal_value)]
            if len(values) > 0 and all(int(v) > 0 for v in values):
                return values
        if not isinstance(shape_expr, ast.Call):
            return None
        if not (
            isinstance(shape_expr.func, ast.Name)
            and str(shape_expr.func.id) == "_resolve_reshape_shape"
            and len(shape_expr.args) >= 2
            and isinstance(shape_expr.args[1], ast.Name)
            and str(shape_expr.args[1].id) == input_name
        ):
            return None
        try:
            raw_new_shape = ast.literal_eval(shape_expr.args[0])
        except Exception:
            return None
        if not isinstance(raw_new_shape, (list, tuple)):
            return None
        raw_values = [int(v) for v in list(raw_new_shape)]
        if len(raw_values) == 0:
            return None
        unknown_index: Optional[int] = None
        known_product = 1
        for index, dim in enumerate(raw_values):
            if int(dim) == -1:
                if unknown_index is not None:
                    return None
                unknown_index = int(index)
                continue
            if int(dim) <= 0:
                return None
            known_product *= int(dim)
        input_product = 1
        for dim in list(input_shape):
            if int(dim) <= 0:
                return None
            input_product *= int(dim)
        resolved = [int(v) for v in raw_values]
        if unknown_index is not None:
            if known_product <= 0 or input_product % known_product != 0:
                return None
            resolved[int(unknown_index)] = int(input_product // known_product)
        if not all(int(v) > 0 for v in resolved):
            return None
        return resolved

    changed = True
    while changed:
        changed = False
        parsed_statements = [ast.parse(line).body[0] for line in rewritten]
        used_names_by_line = [_extract_statement_loads(statement) for statement in parsed_statements]

        for producer_index_in_lines, statement in enumerate(parsed_statements):
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            producer_target = statement.targets[0]
            if not isinstance(producer_target, ast.Name):
                continue
            producer_name = str(producer_target.id)
            producer_reshape = _reshape_call(statement.value)
            if producer_reshape is None:
                continue
            producer_input_expr, producer_shape_expr = producer_reshape
            producer_static_shape = _static_shape_from_expr(
                producer_shape_expr,
                input_name=producer_name,
                input_shape=[],
            )
            if producer_static_shape is None:
                continue
            later_use_lines = [
                int(line_index)
                for line_index in range(int(producer_index_in_lines) + 1, len(used_names_by_line))
                if producer_name in {str(v) for v in used_names_by_line[line_index]}
            ]
            if len(later_use_lines) != 1:
                continue
            consumer_index_in_lines = int(later_use_lines[0])
            if int(consumer_index_in_lines) <= int(producer_index_in_lines):
                continue
            consumer_statement = parsed_statements[int(consumer_index_in_lines)]
            if not isinstance(consumer_statement, ast.Assign) or len(consumer_statement.targets) != 1:
                continue
            consumer_target = consumer_statement.targets[0]
            if not isinstance(consumer_target, ast.Name):
                continue
            consumer_name = str(consumer_target.id)
            consumer_reshape = _reshape_call(consumer_statement.value)
            if consumer_reshape is None:
                continue
            consumer_input_expr, consumer_shape_expr = consumer_reshape
            if not (isinstance(consumer_input_expr, ast.Name) and str(consumer_input_expr.id) == producer_name):
                continue
            final_static_shape = _static_shape_from_expr(
                consumer_shape_expr,
                input_name=producer_name,
                input_shape=producer_static_shape,
            )
            if final_static_shape is None:
                continue
            if not _can_emit_direct_torch_reshape_shape(final_static_shape, allow_zero=False):
                continue
            rewritten[producer_index_in_lines] = (
                f"{consumer_name} = torch.reshape({ast.unparse(producer_input_expr)}, {repr(final_static_shape)})"
            )
            del rewritten[int(consumer_index_in_lines)]
            changed = True
            break
    return rewritten

def _fold_channel_last_affine_conv_bridges(
    lines: Sequence[str],
    *,
    derive_local_var_name: Callable[[str], str],
    channel_first_constant_expr_for_buffer_attr: Callable[[str, Sequence[int]], Optional[str]],
    skipped_module_names: Collection[str] = (),
) -> List[str]:
    if len(lines) < 6:
        return [str(line) for line in lines]
    skipped_module_name_set = {str(name) for name in skipped_module_names}

    materialize_re = re.compile(
        r"^(?P<var>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\((?P<cf>[A-Za-z0-9_]+)\.permute\(0, 2, 3, 1\)\.contiguous\(\), (?P<target>\[[^\]]+\])\)$"
    )
    align_re = re.compile(
        r"^(?P<lhs>[A-Za-z0-9_]+), (?P<rhs>[A-Za-z0-9_]+) = _align_binary_inputs\((?P<input>[A-Za-z0-9_]+), (?P<const>.+), (?P<target>\[[^\]]+\])\)$"
    )
    mul_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.mul\((?P<lhs>[A-Za-z0-9_]+), (?P<rhs>[A-Za-z0-9_]+)\), (?P<target>\[[^\]]+\])\)$"
    )
    add_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.add\((?P<lhs>[A-Za-z0-9_]+), (?P<rhs>[A-Za-z0-9_]+)\), (?P<target>\[[^\]]+\])\)$"
    )
    relu_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = torch\.relu\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    conv_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\)$"
    )

    rewritten: List[str] = []
    index = 0
    while index < len(lines):
        if index + 5 >= len(lines):
            rewritten.extend(str(line) for line in lines[index:])
            break
        materialize_match = materialize_re.match(str(lines[index]))
        if materialize_match is None:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        align1_match = align_re.match(str(lines[index + 1]))
        mul_match = mul_re.match(str(lines[index + 2]))
        align2_match = align_re.match(str(lines[index + 3]))
        add_match = add_re.match(str(lines[index + 4]))
        if (
            align1_match is None
            or mul_match is None
            or align2_match is None
            or add_match is None
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        has_relu = False
        conv_line_index = index + 5
        relu_match = relu_re.match(str(lines[index + 5]))
        if relu_match is not None:
            has_relu = True
            conv_line_index = index + 6
            if conv_line_index >= len(lines):
                rewritten.append(str(lines[index]))
                index += 1
                continue
        conv_match = conv_re.match(str(lines[conv_line_index]))
        if conv_match is None:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if str(conv_match.group("module")) in skipped_module_name_set:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        target_literal = materialize_match.group("target")
        if (
            align1_match.group("input") != materialize_match.group("var")
            or align1_match.group("target") != target_literal
            or mul_match.group("lhs") != align1_match.group("lhs")
            or mul_match.group("rhs") != align1_match.group("rhs")
            or mul_match.group("target") != target_literal
            or align2_match.group("input") != mul_match.group("out")
            or align2_match.group("target") != target_literal
            or add_match.group("lhs") != align2_match.group("lhs")
            or add_match.group("rhs") != align2_match.group("rhs")
            or add_match.group("target") != target_literal
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if has_relu:
            resolved_relu_match = relu_match
            if resolved_relu_match is None:
                rewritten.append(str(lines[index]))
                index += 1
                continue
            if (
                resolved_relu_match.group("input") != add_match.group("out")
                or conv_match.group("input") != resolved_relu_match.group("out")
            ):
                rewritten.append(str(lines[index]))
                index += 1
                continue
        elif conv_match.group("input") != add_match.group("out"):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        try:
            target_shape = ast.literal_eval(target_literal)
        except Exception:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if (
            not isinstance(target_shape, list)
            or len(target_shape) != 4
            or int(target_shape[3]) <= 0
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if not all(token.strip().startswith("self.") for token in [align1_match.group("const"), align2_match.group("const")]):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        channel_count = int(target_shape[3])
        channel_first_shape = [1, channel_count, 1, 1]
        mul_cf_var = derive_local_var_name(f"{mul_match.group('out')}_cf")
        add_cf_var = derive_local_var_name(f"{add_match.group('out')}_cf")
        final_cf_var = add_cf_var
        mul_rhs_expr = channel_first_constant_expr_for_buffer_attr(
            align1_match.group("const"),
            channel_first_shape,
        )
        if mul_rhs_expr is None:
            mul_rhs_expr = f"torch.reshape({align1_match.group('const')}, {repr(channel_first_shape)})"
        add_rhs_expr = channel_first_constant_expr_for_buffer_attr(
            align2_match.group("const"),
            channel_first_shape,
        )
        if add_rhs_expr is None:
            add_rhs_expr = f"torch.reshape({align2_match.group('const')}, {repr(channel_first_shape)})"
        rewritten.append(f"{mul_cf_var} = torch.mul({materialize_match.group('cf')}, {mul_rhs_expr})")
        rewritten.append(f"{add_cf_var} = torch.add({mul_cf_var}, {add_rhs_expr})")
        if has_relu:
            resolved_relu_match = relu_match
            if resolved_relu_match is None:
                rewritten.append(str(lines[index]))
                index += 1
                continue
            final_cf_var = derive_local_var_name(f"{resolved_relu_match.group('out')}_cf")
            rewritten.append(f"{final_cf_var} = torch.relu({add_cf_var})")
        rewritten.append(f"{conv_match.group('out')} = self.{conv_match.group('module')}({final_cf_var})")
        index = conv_line_index + 1
    return rewritten

def _fold_channel_first_gap_conv_bridges(
    lines: Sequence[str],
) -> List[str]:
    mean_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = torch\.mean\((?P<input>[A-Za-z0-9_]+), dim=\[2, 3\], keepdim=True\)$"
    )
    conv_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\)$"
    )

    rewritten = [str(line) for line in lines]
    channel_first_gap_vars: Set[str] = set()
    for index, line in enumerate(rewritten):
        mean_match = mean_re.match(line)
        if mean_match is not None:
            channel_first_gap_vars.add(str(mean_match.group("out")))
            continue
        conv_match = conv_re.match(line)
        if conv_match is None:
            continue
        input_var = str(conv_match.group("input"))
        if input_var not in channel_first_gap_vars:
            continue
        rewritten[index] = (
            f"{conv_match.group('out')} = self.{conv_match.group('module')}({input_var})"
        )
    return rewritten

def _rewrite_channel_first_gap_outputs_to_explicit_channel_last(
    lines: Sequence[str],
) -> List[str]:
    mean_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = torch\.mean\((?P<input>[A-Za-z0-9_]+), dim=\[2, 3\], keepdim=True\)$"
    )
    permute_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = _torch_permute\((?P<input>[A-Za-z0-9_]+), \[0, 2, 3, 1\]\)$"
    )
    rewritten = [str(line) for line in lines]
    for index in range(len(rewritten) - 1):
        mean_match = mean_re.match(rewritten[index])
        if mean_match is None:
            continue
        permute_match = permute_re.match(rewritten[index + 1])
        if permute_match is None:
            continue
        if str(permute_match.group("input")) != str(mean_match.group("out")):
            continue
        rewritten[index] = (
            f"{permute_match.group('out')} = torch.mean("
            f"{mean_match.group('input')}.permute(0, 2, 3, 1).contiguous(), "
            f"dim=[1, 2], keepdim=True)"
        )
        rewritten[index + 1] = ""
    return [line for line in rewritten if line != ""]

def _rewrite_channel_last_gap_means_to_reduce_mean(
    lines: Sequence[str],
) -> List[str]:
    pattern = re.compile(
        r"torch\.mean\((?P<expr>.+?\.permute\(0, 2, 3, 1\)\.contiguous\(\)), dim=\[1, 2\], keepdim=True\)"
    )

    def _rewrite_line(line: str) -> str:
        match = pattern.search(line)
        if match is None:
            return line
        expr = str(match.group("expr"))
        replacement = (
            f"_reduce_mean({expr}, _normalize_axes([1, 2], {expr}.ndim), keepdims=True)"
        )
        return line[: match.start()] + replacement + line[match.end() :]

    return [_rewrite_line(str(line)) for line in lines]

def _fold_boundary_transpose_pad_conv_bridges(
    lines: Sequence[str],
) -> List[str]:
    conv_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0, 2, 3, 1\)\.contiguous\(\)\)$"
    )
    output_bridge_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = _torch_permute\((?P<input>[A-Za-z0-9_]+), \[0, 2, 3, 1\]\)$"
    )
    rewritten = [str(line) for line in lines]
    for index in range(len(rewritten) - 1):
        conv_match = conv_re.match(rewritten[index])
        if conv_match is None:
            continue
        output_bridge_match = output_bridge_re.match(rewritten[index + 1])
        if output_bridge_match is None:
            continue
        if str(output_bridge_match.group("input")) != str(conv_match.group("out")):
            continue
        rewritten[index] = (
            f"{conv_match.group('out')} = self.{conv_match.group('module')}({conv_match.group('input')})"
        )
    return rewritten

def _bridge_boundary_metadata_gather_nd_inputs(
    lines: Sequence[str],
    *,
    model_ir: ModelIR,
    tensor_var_names: Dict[str, str],
) -> List[str]:
    gather_nd_input_perms: Dict[str, List[int]] = {}
    for op in model_ir.operators:
        if str(op.op_type) != "GATHER_ND" or len(op.inputs) < 2 or len(op.outputs) != 1:
            continue
        params_name = str(op.inputs[0])
        if params_name not in model_ir.inputs:
            continue
        params_tensor = model_ir.tensors.get(params_name, None)
        output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if params_tensor is None or output_tensor is None:
            continue
        params_shape = [int(v) for v in list(params_tensor.shape)]
        output_shape = [int(v) for v in list(output_tensor.shape)]
        params_rank = len(params_shape)
        params_layout = normalize_logical_layout(params_tensor.logical_layout)
        if params_rank not in {3, 4, 5} or not is_channel_first_logical_layout(params_layout):
            continue
        cf_to_cl_perm = logical_layout_permutation(
            source_layout=params_layout,
            target_layout=channel_last_logical_layout(params_rank),
        )
        if cf_to_cl_perm is None:
            continue
        actual_gather_nd_shape = _infer_gather_nd_shape_for_codegen(
            model_ir=model_ir,
            params_shape=params_shape,
            indices_tensor_name=str(op.inputs[1]),
        )
        permuted_shape = _permute_shape(params_shape, cf_to_cl_perm)
        permuted_gather_nd_shape = _infer_gather_nd_shape_for_codegen(
            model_ir=model_ir,
            params_shape=permuted_shape,
            indices_tensor_name=str(op.inputs[1]),
        )
        if (
            not _shape_lists_equal(actual_gather_nd_shape, output_shape)
            and _shape_lists_equal(permuted_gather_nd_shape, output_shape)
        ):
            gather_nd_input_perms[str(tensor_var_names[params_name])] = [
                int(v) for v in list(cf_to_cl_perm)
            ]

    if not gather_nd_input_perms:
        return [str(line) for line in lines]

    gather_nd_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = _apply_gather_nd\((?P<input>[A-Za-z0-9_]+), (?P<indices>.+), target_shape=(?P<target>.+)\)$"
    )
    double_perm_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = _apply_gather_nd\(_torch_permute\((?P<input>[A-Za-z0-9_]+), (?P<perm>\[[^\]]+\])\)\.permute\((?P<perm_args>[^\)]+)\)\.contiguous\(\), (?P<indices>.+), target_shape=(?P<target>.+)\)$"
    )
    rewritten = [str(line) for line in lines]
    for index, line in enumerate(rewritten):
        double_perm_match = double_perm_re.match(line)
        if double_perm_match is not None:
            try:
                perm_values = [int(v) for v in ast.literal_eval(double_perm_match.group("perm"))]
                perm_args = [int(v.strip()) for v in str(double_perm_match.group("perm_args")).split(",")]
            except Exception:
                perm_values = []
                perm_args = []
            if perm_values == perm_args:
                rewritten[index] = (
                    f"{double_perm_match.group('out')} = _apply_gather_nd(_torch_permute({double_perm_match.group('input')}, "
                    f"{double_perm_match.group('perm')}), {double_perm_match.group('indices')}, "
                    f"target_shape={double_perm_match.group('target')})"
                )
                continue
        match = gather_nd_re.match(line)
        if match is None:
            continue
        input_var = str(match.group("input"))
        if input_var not in gather_nd_input_perms:
            continue
        perm = gather_nd_input_perms[input_var]
        rewritten[index] = (
            f"{match.group('out')} = _apply_gather_nd(_torch_permute({input_var}, {repr(perm)}), "
            f"{match.group('indices')}, target_shape={match.group('target')})"
        )
    return rewritten

def _collapse_redundant_torch_permute_chains(
    lines: Sequence[str],
) -> List[str]:
    pattern = re.compile(
        r"_torch_permute\((?P<input>[A-Za-z0-9_]+), (?P<perm>\[[^\]]+\])\)\.permute\((?P<perm_args>[^\)]+)\)\.contiguous\(\)"
    )

    def _rewrite_line(line: str) -> str:
        match = pattern.search(line)
        if match is None:
            return line
        try:
            perm_values = [int(v) for v in ast.literal_eval(match.group("perm"))]
            perm_args = [int(v.strip()) for v in str(match.group("perm_args")).split(",")]
        except Exception:
            return line
        if perm_values != perm_args:
            return line
        return (
            line[: match.start()]
            + f"_torch_permute({match.group('input')}, {match.group('perm')})"
            + line[match.end() :]
        )

    return [_rewrite_line(str(line)) for line in lines]

def _inline_trivial_public_layout_bridge_aliases(
    lines: Sequence[str],
) -> List[str]:
    assign_re = re.compile(
        r"^(?P<alias>[A-Za-z0-9_]+_public_layout_bridge) = (?P<source>[A-Za-z0-9_]+)$"
    )
    rewritten = [str(line) for line in lines]
    alias_map: Dict[str, str] = {}
    kept_lines: List[str] = []
    for line in rewritten:
        match = assign_re.match(line)
        if match is not None:
            alias_map[str(match.group("alias"))] = str(match.group("source"))
            continue
        rewritten_line = str(line)
        for alias, source in alias_map.items():
            rewritten_line = re.sub(rf"\b{re.escape(alias)}\b", source, rewritten_line)
        kept_lines.append(rewritten_line)
    return kept_lines

def _fold_channel_last_prelu_bridges(
    lines: Sequence[str],
) -> List[str]:
    in_re = re.compile(
        r"^(?P<var>[A-Za-z0-9_]+) = _torch_permute\((?P<input>[A-Za-z0-9_]+), \[0, 3, 1, 2\]\)$"
    )
    prelu_re = re.compile(
        r"^(?P<var>[A-Za-z0-9_]+) = self\.(?P<module>prelu_[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    out_re = re.compile(
        r"^(?P<var>[A-Za-z0-9_]+) = _torch_permute\((?P<input>[A-Za-z0-9_]+), \[0, 2, 3, 1\]\)$"
    )
    rewritten = [str(line) for line in lines]
    index = 0
    while index <= len(rewritten) - 3:
        in_match = in_re.match(rewritten[index])
        if in_match is None:
            index += 1
            continue
        prelu_match = prelu_re.match(rewritten[index + 1])
        out_match = out_re.match(rewritten[index + 2])
        if prelu_match is None or out_match is None:
            index += 1
            continue
        bridge_var = str(in_match.group("var"))
        if str(prelu_match.group("input")) != bridge_var:
            index += 1
            continue
        prelu_out_var = str(prelu_match.group("var"))
        if str(out_match.group("input")) != prelu_out_var:
            index += 1
            continue
        rewritten[index] = (
            f"{out_match.group('var')} = self.{prelu_match.group('module')}("
            f"{in_match.group('input')}.permute(0, 3, 1, 2).contiguous()"
            f").permute(0, 2, 3, 1).contiguous()"
        )
        rewritten[index + 1] = ""
        rewritten[index + 2] = ""
        index += 3
    return [line for line in rewritten if line != ""]

def _ensure_explicit_depthwise_channel_last_input_bridges(
    lines: Sequence[str],
    *,
    module_names: Collection[str],
) -> List[str]:
    if len(module_names) == 0:
        return [str(line) for line in lines]
    target_module_names = {str(name) for name in module_names}
    module_call_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+) = self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    rewritten: List[str] = []
    for line in lines:
        current_line = str(line)
        match = module_call_re.match(current_line)
        if match is None or str(match.group("module")) not in target_module_names:
            rewritten.append(current_line)
            continue
        input_expr = str(match.group("input"))
        if input_expr.endswith("_cf"):
            bridged_input_expr = (
                f"{input_expr}.permute(0, 2, 3, 1).contiguous()"
                ".permute(0, 3, 1, 2).contiguous()"
            )
        else:
            bridged_input_expr = f"{input_expr}.permute(0, 3, 1, 2).contiguous()"
        rewritten.append(
            f"{match.group('out')} = self.{match.group('module')}({bridged_input_expr})"
        )
    return rewritten

def _build_forward_stage_methods(
    lines: Sequence[str],
    *,
    tensor_var_names: Dict[str, str],
    model_ir: ModelIR,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    if len(lines) < 80:
        return "", [f"        {line}" for line in lines], []

    parsed_statements: List[ast.stmt] = []
    top_level_assigned_names: List[List[str]] = []
    raw_used_names: List[List[str]] = []
    for line in lines:
        statement = ast.parse(str(line)).body[0]
        parsed_statements.append(statement)
        top_level_assigned_names.append(_extract_statement_assignments(statement))
        raw_used_names.append(_extract_statement_loads(statement))

    local_name_candidates: Set[str] = {
        str(tensor_var_names[str(name)]) for name in model_ir.inputs
    }
    local_name_candidates.update(str(tensor_var_names[str(name)]) for name in model_ir.outputs)
    for assigned_names in top_level_assigned_names:
        local_name_candidates.update(str(name) for name in assigned_names)

    assigned_names_by_line: List[List[str]] = []
    used_names_by_line: List[List[str]] = []
    for assigned_names, used_names in zip(top_level_assigned_names, raw_used_names):
        assigned_filtered = [str(name) for name in assigned_names if str(name) in local_name_candidates]
        used_filtered = [str(name) for name in used_names if str(name) in local_name_candidates]
        assigned_names_by_line.append(assigned_filtered)
        used_names_by_line.append(used_filtered)

    tensor_name_by_var_name = {str(var_name): str(tensor_name) for tensor_name, var_name in tensor_var_names.items()}

    def _reshape_call_from_statement(statement: ast.stmt) -> Optional[Tuple[str, str]]:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            return None
        value = statement.value
        if not isinstance(value, ast.Call) or len(value.keywords) != 0 or len(value.args) < 2:
            return None
        func = value.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "reshape"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            return None
        input_expr = value.args[0]
        if not isinstance(input_expr, ast.Name):
            return None
        return str(target.id), str(input_expr.id)

    total_lines = len(lines)

    def _is_adjacent_single_use_static_reshape_chain_boundary(line_index: int) -> bool:
        if int(line_index) < 0 or int(line_index) + 1 >= total_lines:
            return False
        producer_pair = _reshape_call_from_statement(parsed_statements[int(line_index)])
        consumer_pair = _reshape_call_from_statement(parsed_statements[int(line_index) + 1])
        if producer_pair is None or consumer_pair is None:
            return False
        producer_name, _ = producer_pair
        consumer_name, consumer_input_name = consumer_pair
        if consumer_input_name != producer_name:
            return False
        later_use_lines = [
            int(candidate_index)
            for candidate_index in range(int(line_index) + 1, total_lines)
            if producer_name in {str(v) for v in used_names_by_line[candidate_index]}
        ]
        if later_use_lines != [int(line_index) + 1]:
            return False
        consumer_tensor_name = tensor_name_by_var_name.get(consumer_name, None)
        if consumer_tensor_name is None:
            return False
        consumer_tensor = model_ir.tensors.get(str(consumer_tensor_name), None)
        if consumer_tensor is None:
            return False
        preferred_shape = _preferred_reshape_target_values(consumer_tensor)
        if preferred_shape is None:
            preferred_shape = [int(v) for v in list(consumer_tensor.shape)]
        return (
            len(preferred_shape) > 0
            and all(int(v) > 0 for v in list(preferred_shape))
            and _can_emit_direct_torch_reshape_shape(preferred_shape, allow_zero=False)
        )

    final_output_names = {str(tensor_var_names[str(name)]) for name in model_ir.outputs}
    stage_min_lines = 18
    stage_target_lines = 28
    stage_max_lines = 36
    stage_methods: List[str] = []
    forward_stage_calls: List[str] = []
    stage_specs: List[Dict[str, Any]] = []
    stage_index = 0
    start_index = 0

    def _chunk_io(start: int, end: int) -> Tuple[List[str], List[str]]:
        defined: Set[str] = set()
        assigned_order: List[str] = []
        inputs: List[str] = []
        seen_inputs: Set[str] = set()
        for line_index in range(start, end + 1):
            for name in used_names_by_line[line_index]:
                if name not in defined and name not in seen_inputs:
                    seen_inputs.add(name)
                    inputs.append(name)
            for name in assigned_names_by_line[line_index]:
                if name not in defined:
                    defined.add(name)
                    assigned_order.append(name)
        later_needed: Set[str] = set(final_output_names)
        for line_index in range(end + 1, total_lines):
            later_needed.update(used_names_by_line[line_index])
        outputs = [name for name in assigned_order if name in later_needed]
        return inputs, outputs

    def _chunk_io_for_stage_lines(stage_lines_local: Sequence[str], end_index: int) -> Tuple[List[str], List[str]]:
        defined: Set[str] = set()
        assigned_order: List[str] = []
        inputs: List[str] = []
        seen_inputs: Set[str] = set()
        stage_statements = [ast.parse(line).body[0] for line in stage_lines_local]
        for statement in stage_statements:
            for name in [str(v) for v in _extract_statement_loads(statement) if str(v) in local_name_candidates]:
                if name not in defined and name not in seen_inputs:
                    seen_inputs.add(name)
                    inputs.append(name)
            for name in [str(v) for v in _extract_statement_assignments(statement) if str(v) in local_name_candidates]:
                if name not in defined:
                    defined.add(name)
                    assigned_order.append(name)
        later_needed: Set[str] = set(final_output_names)
        for line_index in range(end_index + 1, total_lines):
            later_needed.update(used_names_by_line[line_index])
        outputs = [name for name in assigned_order if name in later_needed]
        return inputs, outputs

    def _append_stage(start: int, end: int) -> None:
        nonlocal stage_index
        stage_lines = _fold_single_use_static_reshape_chains(
            lines[start:end + 1],
            tensor_var_names=tensor_var_names,
            model_ir=model_ir,
        )
        stage_inputs, stage_outputs = _chunk_io_for_stage_lines(stage_lines, end)
        if len(stage_outputs) == 0:
            forward_stage_calls.extend(f"        {line}" for line in stage_lines)
            return
        method_name = f"_forward_stage_{stage_index}"
        arg_list = ", ".join(f"{name}: torch.Tensor" for name in stage_inputs)
        signature = f"    def {method_name}(self, {arg_list})" if len(arg_list) > 0 else f"    def {method_name}(self)"
        if len(stage_outputs) == 1:
            signature += " -> torch.Tensor:\n"
        else:
            signature += " -> tuple[" + ", ".join("torch.Tensor" for _ in stage_outputs) + "]:\n"
        stage_body = "\n".join(f"        {line}" for line in stage_lines)
        if len(stage_outputs) == 1:
            stage_return = f"        return {stage_outputs[0]}\n"
        else:
            stage_return = f"        return ({', '.join(stage_outputs)})\n"
        stage_methods.append(f"{signature}{stage_body}\n{stage_return}")

        call_args = ", ".join(stage_inputs)
        call_expr = f"self.{method_name}({call_args})" if len(call_args) > 0 else f"self.{method_name}()"
        if len(stage_outputs) == 1:
            forward_stage_calls.append(f"        {stage_outputs[0]} = {call_expr}")
        else:
            forward_stage_calls.append(f"        {', '.join(stage_outputs)} = {call_expr}")
        stage_specs.append(
            {
                "stage_index": int(stage_index),
                "method_name": str(method_name),
                "inputs": list(stage_inputs),
                "outputs": list(stage_outputs),
            }
        )
        stage_index += 1

    while start_index < total_lines:
        remaining = total_lines - start_index
        if remaining < 80 or remaining <= stage_max_lines:
            _append_stage(start_index, total_lines - 1)
            break
        candidate_min_end = start_index + stage_min_lines - 1
        candidate_max_end = min(start_index + stage_max_lines - 1, total_lines - stage_min_lines - 1)
        if candidate_min_end > candidate_max_end:
            _append_stage(start_index, total_lines - 1)
            break
        best_candidate: Optional[Tuple[int, List[str], List[str], Tuple[int, int, int]]] = None
        for end_index in range(candidate_min_end, candidate_max_end + 1):
            if _is_adjacent_single_use_static_reshape_chain_boundary(end_index):
                continue
            inputs, outputs = _chunk_io(start_index, end_index)
            if len(outputs) == 0:
                continue
            score = (
                len(inputs) + len(outputs),
                abs((end_index - start_index + 1) - stage_target_lines),
                len(outputs),
            )
            if best_candidate is None or score < best_candidate[3]:
                best_candidate = (end_index, inputs, outputs, score)
        if best_candidate is None:
            _append_stage(start_index, total_lines - 1)
            break
        end_index, _, _, _ = best_candidate
        _append_stage(start_index, end_index)
        start_index = end_index + 1

    stage_methods_source = "\n".join(stage_methods)
    if len(stage_methods_source) > 0:
        stage_methods_source += "\n"
    return stage_methods_source, forward_stage_calls, stage_specs

def _rewrite_channel_last_binary_bridge_chains(
    lines: Sequence[str],
    *,
    derive_local_var_name: Callable[[str], str],
    channel_first_constant_expr_for_buffer_attr: Callable[[str, Sequence[int]], Optional[str]],
    skipped_module_names: Collection[str] = (),
) -> List[str]:
    rewritten: List[str] = []
    line_count = len(lines)
    index = 0
    skipped_module_name_set = {str(name) for name in skipped_module_names}
    binary_fn_pattern = r"(?:add|sub|mul|div|maximum|minimum)"
    public_bridge_pattern = re.compile(
        r"^(?P<bridge>\w+)\s*=\s*_torch_permute\((?P<input>\w+), \[0, 2, 3, 1\]\)$"
    )
    binary_align_pattern = re.compile(
        rf"^(?P<out>\w+)\s*=\s*_align_tensor_to_target_shape\(torch\.(?P<fn>{binary_fn_pattern})\((?P<lhs>[^,]+), (?P<rhs>[^,]+)\), (?P<shape>\[[^\]]+\])\)$"
    )
    conv_input_bridge_pattern = re.compile(
        r"^(?P<out>\w+)\s*=\s*self\.(?P<module>\w+)\((?P<input>\w+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\)$"
    )
    output_bridge_pattern = re.compile(
        r"^(?P<out>\w+)\s*=\s*_align_tensor_to_target_shape\((?P<input>\w+)\.permute\(0, 2, 3, 1\)\.contiguous\(\), (?P<shape>\[[^\]]+\])\)$"
    )
    transpose_back_pattern = re.compile(
        r"^(?P<out>\w+)\s*=\s*_torch_permute\((?P<input>\w+), \[0, 3, 1, 2\]\)$"
    )

    def _name_use_count(candidate_lines: Sequence[str], name: str, *, start: int) -> int:
        pattern = re.compile(rf"\b{re.escape(str(name))}\b")
        return sum(1 for line in candidate_lines[start:] if pattern.search(str(line)))

    while index < line_count:
        if index + 2 < line_count:
            bridge_match = public_bridge_pattern.match(str(lines[index]))
            binary_match = binary_align_pattern.match(str(lines[index + 1]))
            conv_match = conv_input_bridge_pattern.match(str(lines[index + 2]))
            if (
                bridge_match is not None
                and binary_match is not None
                and conv_match is not None
                and str(conv_match.group("module")) not in skipped_module_name_set
                and binary_match.group("out") == conv_match.group("input")
                and _name_use_count(lines, bridge_match.group("bridge"), start=index + 1) == 1
                and _name_use_count(lines, binary_match.group("out"), start=index + 2) == 1
            ):
                bridge_var = bridge_match.group("bridge")
                source_var = bridge_match.group("input")
                lhs_expr = binary_match.group("lhs").strip()
                rhs_expr = binary_match.group("rhs").strip()
                target_shape = ast.literal_eval(binary_match.group("shape"))
                if (
                    isinstance(target_shape, list)
                    and len(target_shape) == 4
                    and all(isinstance(v, int) for v in list(target_shape))
                ):
                    cf_constant_shape = [int(target_shape[0]), int(target_shape[3]), 1, 1]
                    buffer_expr = None
                    if lhs_expr == bridge_var and rhs_expr.startswith("self."):
                        buffer_expr = rhs_expr
                    elif rhs_expr == bridge_var and lhs_expr.startswith("self."):
                        buffer_expr = lhs_expr
                    if buffer_expr is not None:
                        cf_constant_expr = channel_first_constant_expr_for_buffer_attr(
                            buffer_expr,
                            cf_constant_shape,
                        )
                        if cf_constant_expr is not None:
                            binary_cf_var = derive_local_var_name(f"{binary_match.group('out')}_cf")
                            cf_lhs = source_var if lhs_expr == bridge_var else cf_constant_expr
                            cf_rhs = source_var if rhs_expr == bridge_var else cf_constant_expr
                            rewritten.append(
                                f"{binary_cf_var} = torch.{binary_match.group('fn')}({cf_lhs}, {cf_rhs})"
                            )
                            rewritten.append(
                                f"{conv_match.group('out')} = self.{conv_match.group('module')}({binary_cf_var})"
                            )
                            index += 3
                            continue
        if index + 3 < line_count:
            output_bridge_match = output_bridge_pattern.match(str(lines[index + 1]))
            binary_match = binary_align_pattern.match(str(lines[index + 2]))
            transpose_back_match = transpose_back_pattern.match(str(lines[index + 3]))
            current_line = str(lines[index])
            if (
                output_bridge_match is not None
                and binary_match is not None
                and transpose_back_match is not None
                and output_bridge_match.group("out") in {
                    binary_match.group("lhs").strip(),
                    binary_match.group("rhs").strip(),
                }
                and binary_match.group("out") == transpose_back_match.group("input")
                and _name_use_count(lines, output_bridge_match.group("out"), start=index + 2) == 1
                and _name_use_count(lines, binary_match.group("out"), start=index + 3) == 1
            ):
                target_shape = ast.literal_eval(output_bridge_match.group("shape"))
                if (
                    isinstance(target_shape, list)
                    and len(target_shape) == 4
                    and all(isinstance(v, int) for v in list(target_shape))
                ):
                    cf_constant_shape = [int(target_shape[0]), int(target_shape[3]), 1, 1]
                    bridge_var = output_bridge_match.group("out")
                    cf_source_var = output_bridge_match.group("input")
                    lhs_expr = binary_match.group("lhs").strip()
                    rhs_expr = binary_match.group("rhs").strip()
                    buffer_expr = None
                    if lhs_expr == bridge_var and rhs_expr.startswith("self."):
                        buffer_expr = rhs_expr
                    elif rhs_expr == bridge_var and lhs_expr.startswith("self."):
                        buffer_expr = lhs_expr
                    if buffer_expr is not None:
                        cf_constant_expr = channel_first_constant_expr_for_buffer_attr(
                            buffer_expr,
                            cf_constant_shape,
                        )
                        if cf_constant_expr is not None:
                            rewritten.append(current_line)
                            cf_lhs = cf_source_var if lhs_expr == bridge_var else cf_constant_expr
                            cf_rhs = cf_source_var if rhs_expr == bridge_var else cf_constant_expr
                            rewritten.append(
                                f"{transpose_back_match.group('out')} = torch.{binary_match.group('fn')}({cf_lhs}, {cf_rhs})"
                            )
                            index += 4
                            continue
        rewritten.append(str(lines[index]))
        index += 1
    return rewritten

def _fold_rank4_reshape_permute_conv_bridges(
    lines: Sequence[str],
) -> List[str]:
    if len(lines) < 3:
        return [str(line) for line in lines]

    rewritten: List[str] = []
    reshape_re = re.compile(
        r"^(?P<out>\w+)\s*=\s*torch\.reshape\((?P<input>[^,]+), (?P<shape>\[[^\]]+\])\)$"
    )
    nhwc_bridge_re = re.compile(
        r"^(?P<out>\w+)\s*=\s*torch\.reshape\((?P<input>\w+)\.permute\(0, 2, 3, 1\)\.contiguous\(\), (?P<shape>\[[^\]]+\])\)$"
    )
    conv_bridge_re = re.compile(
        r"^(?P<out>\w+)\s*=\s*_align_tensor_to_target_shape\(self\.(?P<module>\w+)\((?P<input>\w+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\), (?P<target>\[[^\]]+\])\)$"
    )

    index = 0
    while index < len(lines):
        if index + 2 >= len(lines):
            rewritten.extend(str(line) for line in lines[index:])
            break
        reshape_match = reshape_re.match(str(lines[index]))
        bridge_match = nhwc_bridge_re.match(str(lines[index + 1]))
        conv_match = conv_bridge_re.match(str(lines[index + 2]))
        if (
            reshape_match is None
            or bridge_match is None
            or conv_match is None
            or bridge_match.group("input") != reshape_match.group("out")
            or conv_match.group("input") != bridge_match.group("out")
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        try:
            reshape_shape = ast.literal_eval(reshape_match.group("shape"))
            bridge_shape = ast.literal_eval(bridge_match.group("shape"))
        except Exception:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if (
            not isinstance(reshape_shape, list)
            or not isinstance(bridge_shape, list)
            or len(reshape_shape) != 4
            or len(bridge_shape) != 4
            or _permute_shape(reshape_shape, [0, 2, 3, 1]) != [int(v) for v in list(bridge_shape)]
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        rewritten.append(str(lines[index]))
        rewritten.append(
            f"{conv_match.group('out')} = _align_tensor_to_target_shape(self.{conv_match.group('module')}({reshape_match.group('out')}), {conv_match.group('target')})"
        )
        index += 3
    return rewritten

def _build_named_encoder_methods(
    stage_specs: Sequence[Dict[str, Any]],
    *,
    final_output_names: Set[str],
) -> Tuple[str, List[str], List[str]]:
    if len(stage_specs) == 0:
        return "", [], []

    def _call_line_from_spec(spec: Dict[str, Any]) -> str:
        outputs = [str(name) for name in list(spec.get("outputs", []))]
        call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
        call_expr = f"self.{spec['method_name']}({call_args})" if len(call_args) > 0 else f"self.{spec['method_name']}()"
        if len(outputs) == 1:
            return f"        {outputs[0]} = {call_expr}"
        return f"        {', '.join(outputs)} = {call_expr}"

    layer_pattern = re.compile(r"bert_encoder_layer_(\d+)", flags=re.IGNORECASE)
    default_forward_lines = [_call_line_from_spec(spec) for spec in stage_specs]
    if not any(
        any(
            layer_pattern.search(str(name)) is not None
            and "attention_self_mul_1" not in str(name)
            for name in list(spec.get("outputs", []))
        )
        for spec in stage_specs
    ):
        return "", [], default_forward_lines

    def _stage_layer_index(spec: Dict[str, Any]) -> Optional[int]:
        matches: List[int] = []
        for name in list(spec.get("outputs", [])):
            if "attention_self_mul_1" in str(name):
                continue
            match = layer_pattern.search(str(name))
            if match is not None:
                matches.append(int(match.group(1)))
        if len(matches) == 0:
            return None
        return min(matches)

    grouped_ranges: List[Tuple[int, int, int]] = []
    start_spec_index: Optional[int] = None
    active_layer_index: Optional[int] = None
    for spec_index, spec in enumerate(stage_specs):
        layer_index = _stage_layer_index(spec)
        if layer_index is None:
            if start_spec_index is not None and active_layer_index is not None:
                grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(spec_index - 1)))
                start_spec_index = None
                active_layer_index = None
            continue
        if start_spec_index is None:
            start_spec_index = int(spec_index)
            active_layer_index = int(layer_index)
            continue
        if active_layer_index is None:
            active_layer_index = int(layer_index)
            continue
        if int(layer_index) != int(active_layer_index):
            grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(spec_index - 1)))
            start_spec_index = int(spec_index)
            active_layer_index = int(layer_index)
    if start_spec_index is not None and active_layer_index is not None:
        grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(len(stage_specs) - 1)))

    if len(grouped_ranges) == 0:
        return "", [], default_forward_lines

    class_chunks: List[str] = []
    init_lines: List[str] = []
    forward_lines_local: List[str] = []
    previous_end = 0

    def _group_io(start_idx: int, end_idx: int) -> Tuple[List[str], List[str]]:
        defined: Set[str] = set()
        assigned_order: List[str] = []
        seen_inputs: Set[str] = set()
        method_inputs: List[str] = []
        for spec_index in range(start_idx, end_idx + 1):
            spec = stage_specs[spec_index]
            for name in list(spec.get("inputs", [])):
                normalized = str(name)
                if normalized not in defined and normalized not in seen_inputs:
                    seen_inputs.add(normalized)
                    method_inputs.append(normalized)
            for name in list(spec.get("outputs", [])):
                normalized = str(name)
                if normalized not in defined:
                    defined.add(normalized)
                    assigned_order.append(normalized)
        later_needed = set(final_output_names)
        for spec_index in range(end_idx + 1, len(stage_specs)):
            later_needed.update(str(name) for name in list(stage_specs[spec_index].get("inputs", [])))
        method_outputs = [name for name in assigned_order if name in later_needed]
        return method_inputs, method_outputs

    def _emit_submodule_class_from_stage_range(
        *,
        class_name: str,
        start_idx: int,
        end_idx: int,
    ) -> Optional[Tuple[str, List[str], List[str], List[str]]]:
        method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
        if len(method_outputs) == 0:
            return None
        class_stage_specs = list(stage_specs[start_idx:end_idx + 1])
        init_signature_lines = [
            f"class {class_name}(torch.nn.Module):",
            "    def __init__(",
            "        self,",
            "        *,",
        ]
        init_body_lines = ["        super().__init__()"]
        for spec in class_stage_specs:
            stage_name = str(spec["method_name"])
            init_signature_lines.append(f"        {stage_name}: Callable[..., Any],")
            init_body_lines.append(f"        self.{stage_name} = {stage_name}")
        init_signature_lines.append("    ) -> None:")
        arg_list = ", ".join(f"{name}: torch.Tensor" for name in method_inputs)
        signature = "    def forward(self"
        if len(arg_list) > 0:
            signature += f", {arg_list}"
        signature += ")"
        if len(method_outputs) == 1:
            signature += " -> torch.Tensor:\n"
        else:
            signature += " -> tuple[" + ", ".join("torch.Tensor" for _ in method_outputs) + "]:\n"
        method_body_lines: List[str] = []
        init_call_lines: List[str] = []
        for spec in class_stage_specs:
            outputs = [str(name) for name in list(spec.get("outputs", []))]
            call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
            call_expr = f"self.{spec['method_name']}({call_args})" if len(call_args) > 0 else f"self.{spec['method_name']}()"
            if len(outputs) == 1:
                method_body_lines.append(f"        {outputs[0]} = {call_expr}")
            else:
                method_body_lines.append(f"        {', '.join(outputs)} = {call_expr}")
            init_call_lines.append(f"            {spec['method_name']}=self.{spec['method_name']},")
        if len(method_outputs) == 1:
            return_line = f"        return {method_outputs[0]}\n"
        else:
            return_line = f"        return ({', '.join(method_outputs)})\n"
        class_source = (
            "\n".join(init_signature_lines)
            + "\n"
            + "\n".join(init_body_lines)
            + "\n\n"
            + f"{signature}"
            + "\n".join(method_body_lines)
            + "\n"
            + return_line
        )
        return class_source, method_inputs, method_outputs, init_call_lines

    for layer_index, start_idx, end_idx in grouped_ranges:
        while previous_end < start_idx:
            spec = stage_specs[previous_end]
            forward_lines_local.append(_call_line_from_spec(spec))
            previous_end += 1

        method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
        if len(method_outputs) == 0:
            for spec_index in range(start_idx, end_idx + 1):
                spec = stage_specs[spec_index]
                forward_lines_local.append(_call_line_from_spec(spec))
            previous_end = int(end_idx + 1)
            continue

        method_name = f"_forward_encoder_layer_{int(layer_index)}"
        layer_prefix = f"bert_encoder_layer_{int(layer_index)}_"
        split_idx: Optional[int] = None
        for spec_index in range(start_idx, end_idx + 1):
            output_names = [str(name) for name in list(stage_specs[spec_index].get("outputs", []))]
            if any(
                output_name.startswith(layer_prefix + marker)
                for output_name in output_names
                for marker in ("ffn_", "output_", "output_bottleneck_")
            ):
                split_idx = int(spec_index)
                break

        composite_init_lines: Optional[List[str]] = None
        layer_attr_name = f"encoder_layer_{int(layer_index)}"
        if split_idx is not None and int(split_idx) > int(start_idx):
            attention_class_name = f"_GeneratedEncoderLayer{int(layer_index)}Attention"
            ffn_class_name = f"_GeneratedEncoderLayer{int(layer_index)}Ffn"
            attention_result = _emit_submodule_class_from_stage_range(
                class_name=attention_class_name,
                start_idx=int(start_idx),
                end_idx=int(split_idx - 1),
            )
            ffn_result = _emit_submodule_class_from_stage_range(
                class_name=ffn_class_name,
                start_idx=int(split_idx),
                end_idx=int(end_idx),
            )
            if attention_result is not None and ffn_result is not None:
                attention_source, attention_inputs, attention_outputs, attention_init_calls = attention_result
                ffn_source, _, ffn_outputs, ffn_init_calls = ffn_result
                class_chunks.extend([attention_source, "", ffn_source, ""])
                attention_attr_name = f"{layer_attr_name}_attention"
                ffn_attr_name = f"{layer_attr_name}_ffn"
                init_lines.append(f"self.{attention_attr_name} = {attention_class_name}(")
                init_lines.extend(attention_init_calls)
                init_lines.append("        )")
                init_lines.append(f"self.{ffn_attr_name} = {ffn_class_name}(")
                init_lines.extend(ffn_init_calls)
                init_lines.append("        )")
                composite_init_lines = [
                    f"self.{layer_attr_name} = torch.nn.ModuleDict({{",
                    f"            'attention': self.{attention_attr_name},",
                    f"            'ffn': self.{ffn_attr_name},",
                    "        })",
                ]
                attention_call_args = ", ".join(attention_inputs)
                attention_call_expr = (
                    f"self.{layer_attr_name}['attention']({attention_call_args})"
                    if len(attention_call_args) > 0
                    else f"self.{layer_attr_name}['attention']()"
                )
                if len(attention_outputs) == 1:
                    forward_lines_local.append(f"        {attention_outputs[0]} = {attention_call_expr}")
                else:
                    forward_lines_local.append(
                        f"        {', '.join(attention_outputs)} = {attention_call_expr}"
                    )
                ffn_call_args = ", ".join(attention_outputs)
                ffn_call_expr = (
                    f"self.{layer_attr_name}['ffn']({ffn_call_args})"
                    if len(ffn_call_args) > 0
                    else f"self.{layer_attr_name}['ffn']()"
                )
                if len(ffn_outputs) == 1:
                    forward_lines_local.append(f"        {ffn_outputs[0]} = {ffn_call_expr}")
                else:
                    forward_lines_local.append(f"        {', '.join(ffn_outputs)} = {ffn_call_expr}")
                previous_end = int(end_idx + 1)
                if composite_init_lines is not None:
                    init_lines.extend(composite_init_lines)
                continue

        class_name = f"_GeneratedEncoderLayer{int(layer_index)}"
        emitted = _emit_submodule_class_from_stage_range(
            class_name=class_name,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
        )
        if emitted is None:
            for spec_index in range(start_idx, end_idx + 1):
                spec = stage_specs[spec_index]
                forward_lines_local.append(_call_line_from_spec(spec))
            previous_end = int(end_idx + 1)
            continue
        class_source, _, method_outputs, init_call_lines = emitted
        class_chunks.extend([class_source, ""])
        init_lines.append(f"self.{layer_attr_name} = {class_name}(")
        init_lines.extend(init_call_lines)
        init_lines.append("        )")
        call_expr = f"self.{layer_attr_name}({', '.join(method_inputs)})" if len(method_inputs) > 0 else f"self.{layer_attr_name}()"
        if len(method_outputs) == 1:
            forward_lines_local.append(f"        {method_outputs[0]} = {call_expr}")
        else:
            forward_lines_local.append(f"        {', '.join(method_outputs)} = {call_expr}")
        previous_end = int(end_idx + 1)

    while previous_end < len(stage_specs):
        spec = stage_specs[previous_end]
        forward_lines_local.append(_call_line_from_spec(spec))
        previous_end += 1

    named_class_source = "\n".join(class_chunks)
    if len(named_class_source) > 0:
        named_class_source += "\n"
    return named_class_source, init_lines, forward_lines_local

def _build_named_encoder_methods_composite(
    stage_specs: Sequence[Dict[str, Any]],
    *,
    final_output_names: Set[str],
) -> Tuple[str, List[str], List[str]]:
    if len(stage_specs) == 0:
        return "", [], []

    def _call_line_from_spec(spec: Dict[str, Any]) -> str:
        outputs = [str(name) for name in list(spec.get("outputs", []))]
        call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
        call_expr = f"self.{spec['method_name']}({call_args})" if len(call_args) > 0 else f"self.{spec['method_name']}()"
        if len(outputs) == 1:
            return f"        {outputs[0]} = {call_expr}"
        return f"        {', '.join(outputs)} = {call_expr}"

    layer_pattern = re.compile(r"bert_encoder_layer_(\d+)", flags=re.IGNORECASE)
    default_forward_lines = [_call_line_from_spec(spec) for spec in stage_specs]
    if not any(
        any(
            layer_pattern.search(str(name)) is not None
            and "attention_self_mul_1" not in str(name)
            for name in list(spec.get("outputs", []))
        )
        for spec in stage_specs
    ):
        return "", [], default_forward_lines

    def _stage_layer_index(spec: Dict[str, Any]) -> Optional[int]:
        matches: List[int] = []
        for name in list(spec.get("outputs", [])):
            if "attention_self_mul_1" in str(name):
                continue
            match = layer_pattern.search(str(name))
            if match is not None:
                matches.append(int(match.group(1)))
        if len(matches) == 0:
            return None
        return min(matches)

    grouped_ranges: List[Tuple[int, int, int]] = []
    start_spec_index: Optional[int] = None
    active_layer_index: Optional[int] = None
    for spec_index, spec in enumerate(stage_specs):
        layer_index = _stage_layer_index(spec)
        if layer_index is None:
            if start_spec_index is not None and active_layer_index is not None:
                grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(spec_index - 1)))
                start_spec_index = None
                active_layer_index = None
            continue
        if start_spec_index is None:
            start_spec_index = int(spec_index)
            active_layer_index = int(layer_index)
            continue
        if active_layer_index is None:
            active_layer_index = int(layer_index)
            continue
        if int(layer_index) != int(active_layer_index):
            grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(spec_index - 1)))
            start_spec_index = int(spec_index)
            active_layer_index = int(layer_index)
    if start_spec_index is not None and active_layer_index is not None:
        grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(len(stage_specs) - 1)))

    if len(grouped_ranges) == 0:
        return "", [], default_forward_lines

    class_chunks: List[str] = []
    init_lines: List[str] = []
    forward_lines_local: List[str] = []
    previous_end = 0

    def _group_io(start_idx: int, end_idx: int) -> Tuple[List[str], List[str]]:
        defined: Set[str] = set()
        assigned_order: List[str] = []
        seen_inputs: Set[str] = set()
        method_inputs: List[str] = []
        for spec_index in range(start_idx, end_idx + 1):
            spec = stage_specs[spec_index]
            for name in list(spec.get("inputs", [])):
                normalized = str(name)
                if normalized not in defined and normalized not in seen_inputs:
                    seen_inputs.add(normalized)
                    method_inputs.append(normalized)
            for name in list(spec.get("outputs", [])):
                normalized = str(name)
                if normalized not in defined:
                    defined.add(normalized)
                    assigned_order.append(normalized)
        later_needed = set(final_output_names)
        for spec_index in range(end_idx + 1, len(stage_specs)):
            later_needed.update(str(name) for name in list(stage_specs[spec_index].get("inputs", [])))
        method_outputs = [name for name in assigned_order if name in later_needed]
        return method_inputs, method_outputs

    def _emit_submodule_class_from_stage_range(
        *,
        class_name: str,
        start_idx: int,
        end_idx: int,
    ) -> Optional[Tuple[str, List[str], List[str], List[str]]]:
        method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
        if len(method_outputs) == 0:
            return None
        class_stage_specs = list(stage_specs[start_idx:end_idx + 1])
        init_signature_lines = [
            f"class {class_name}(torch.nn.Module):",
            "    def __init__(",
            "        self,",
            "        *,",
        ]
        init_body_lines = ["        super().__init__()"]
        for spec in class_stage_specs:
            stage_name = str(spec["method_name"])
            init_signature_lines.append(f"        {stage_name}: Callable[..., Any],")
            init_body_lines.append(f"        self.{stage_name} = {stage_name}")
        init_signature_lines.append("    ) -> None:")
        arg_list = ", ".join(f"{name}: torch.Tensor" for name in method_inputs)
        signature = "    def forward(self"
        if len(arg_list) > 0:
            signature += f", {arg_list}"
        signature += ")"
        if len(method_outputs) == 1:
            signature += " -> torch.Tensor:\n"
        else:
            signature += " -> tuple[" + ", ".join("torch.Tensor" for _ in method_outputs) + "]:\n"
        method_body_lines: List[str] = []
        init_call_lines: List[str] = []
        for spec in class_stage_specs:
            outputs = [str(name) for name in list(spec.get("outputs", []))]
            call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
            call_expr = f"self.{spec['method_name']}({call_args})" if len(call_args) > 0 else f"self.{spec['method_name']}()"
            if len(outputs) == 1:
                method_body_lines.append(f"        {outputs[0]} = {call_expr}")
            else:
                method_body_lines.append(f"        {', '.join(outputs)} = {call_expr}")
            init_call_lines.append(f"            {spec['method_name']}=self.{spec['method_name']},")
        if len(method_outputs) == 1:
            return_line = f"        return {method_outputs[0]}\n"
        else:
            return_line = f"        return ({', '.join(method_outputs)})\n"
        class_source = (
            "\n".join(init_signature_lines)
            + "\n"
            + "\n".join(init_body_lines)
            + "\n\n"
            + f"{signature}"
            + "\n".join(method_body_lines)
            + "\n"
            + return_line
        )
        return class_source, method_inputs, method_outputs, init_call_lines

    for layer_index, start_idx, end_idx in grouped_ranges:
        while previous_end < start_idx:
            spec = stage_specs[previous_end]
            forward_lines_local.append(_call_line_from_spec(spec))
            previous_end += 1

        method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
        if len(method_outputs) == 0:
            for spec_index in range(start_idx, end_idx + 1):
                spec = stage_specs[spec_index]
                forward_lines_local.append(_call_line_from_spec(spec))
            previous_end = int(end_idx + 1)
            continue

        layer_prefix = f"bert_encoder_layer_{int(layer_index)}_"
        split_idx: Optional[int] = None
        for spec_index in range(start_idx, end_idx + 1):
            output_names = [str(name) for name in list(stage_specs[spec_index].get("outputs", []))]
            if any(
                output_name.startswith(layer_prefix + marker)
                for output_name in output_names
                for marker in ("ffn_", "output_", "output_bottleneck_")
            ):
                split_idx = int(spec_index)
                break

        composite_init_lines: Optional[List[str]] = None
        layer_attr_name = f"encoder_layer_{int(layer_index)}"
        if split_idx is not None and int(split_idx) > int(start_idx):
            attention_class_name = f"_GeneratedEncoderLayer{int(layer_index)}Attention"
            ffn_class_name = f"_GeneratedEncoderLayer{int(layer_index)}FFN"
            layer_class_name = f"_GeneratedEncoderLayer{int(layer_index)}"
            attention_emitted = _emit_submodule_class_from_stage_range(
                class_name=attention_class_name,
                start_idx=int(start_idx),
                end_idx=int(split_idx - 1),
            )
            ffn_emitted = _emit_submodule_class_from_stage_range(
                class_name=ffn_class_name,
                start_idx=int(split_idx),
                end_idx=int(end_idx),
            )
            if attention_emitted is not None and ffn_emitted is not None:
                attention_source, attention_inputs, attention_outputs, attention_init_call_lines = attention_emitted
                ffn_source, ffn_inputs, ffn_outputs, ffn_init_call_lines = ffn_emitted
                class_chunks.append(attention_source)
                class_chunks.append(ffn_source)
                class_chunks.append(
                    "class {layer_class_name}(torch.nn.Module):\n"
                    "    def __init__(self, *, attention: torch.nn.Module, ffn: torch.nn.Module) -> None:\n"
                    "        super().__init__()\n"
                    "        self.attention = attention\n"
                    "        self.ffn = ffn\n\n"
                    "    def forward({signature_args}){signature_return}"
                    "{body}"
                    "{return_line}".format(
                        layer_class_name=layer_class_name,
                        signature_args=(
                            "self"
                            + (", " + ", ".join(f"{name}: torch.Tensor" for name in method_inputs) if len(method_inputs) > 0 else "")
                        ),
                        signature_return=(
                            " -> torch.Tensor:\n"
                            if len(method_outputs) == 1 else
                            " -> tuple[" + ", ".join("torch.Tensor" for _ in method_outputs) + "]:\n"
                        ),
                        body="".join(
                            [
                                (
                                    f"        {attention_outputs[0]} = self.attention({', '.join(attention_inputs)})\n"
                                    if len(attention_outputs) == 1 else
                                    f"        {', '.join(attention_outputs)} = self.attention({', '.join(attention_inputs)})\n"
                                ),
                                (
                                    f"        {ffn_outputs[0]} = self.ffn({', '.join(ffn_inputs)})\n"
                                    if len(ffn_outputs) == 1 else
                                    f"        {', '.join(ffn_outputs)} = self.ffn({', '.join(ffn_inputs)})\n"
                                ),
                            ]
                        ),
                        return_line=(
                            f"        return {method_outputs[0]}\n"
                            if len(method_outputs) == 1 else
                            f"        return ({', '.join(method_outputs)})\n"
                        ),
                    )
                )
                composite_init_lines = [
                    f"self.{layer_attr_name} = {layer_class_name}(",
                    f"    attention={attention_class_name}(",
                    *attention_init_call_lines,
                    "    ),",
                    f"    ffn={ffn_class_name}(",
                    *ffn_init_call_lines,
                    "    ),",
                    ")",
                ]

        if composite_init_lines is None:
            layer_class_name = f"_GeneratedEncoderLayer{int(layer_index)}"
            emitted = _emit_submodule_class_from_stage_range(
                class_name=layer_class_name,
                start_idx=int(start_idx),
                end_idx=int(end_idx),
            )
            if emitted is None:
                for spec_index in range(start_idx, end_idx + 1):
                    spec = stage_specs[spec_index]
                    forward_lines_local.append(_call_line_from_spec(spec))
                previous_end = int(end_idx + 1)
                continue
            layer_source, _, _, layer_init_call_lines = emitted
            class_chunks.append(layer_source)
            composite_init_lines = [
                f"self.{layer_attr_name} = {layer_class_name}(",
                *layer_init_call_lines,
                ")",
            ]
        if composite_init_lines is not None:
            init_lines.extend(composite_init_lines)

        call_args = ", ".join(method_inputs)
        call_expr = f"self.{layer_attr_name}({call_args})" if len(call_args) > 0 else f"self.{layer_attr_name}()"
        if len(method_outputs) == 1:
            forward_lines_local.append(f"        {method_outputs[0]} = {call_expr}")
        else:
            forward_lines_local.append(f"        {', '.join(method_outputs)} = {call_expr}")
        previous_end = int(end_idx + 1)

    while previous_end < len(stage_specs):
        spec = stage_specs[previous_end]
        forward_lines_local.append(_call_line_from_spec(spec))
        previous_end += 1

    named_class_source = "\n".join(class_chunks)
    if len(named_class_source) > 0:
        named_class_source += "\n"
    return named_class_source, init_lines, forward_lines_local

def _sequence_lstm_input_name(op: OperatorIR, index: int) -> str:
    if int(index) < 0 or int(index) >= len(op.inputs):
        return ""
    return str(op.inputs[int(index)]).strip()

def _tensor_has_constant_data(model_ir: ModelIR, tensor_name: str) -> bool:
    if str(tensor_name).strip() == "":
        return False
    tensor = model_ir.tensors.get(str(tensor_name), None)
    return tensor is not None and isinstance(tensor.data, np.ndarray)

def _sequence_lstm_bias_inputs_supported(
    model_ir: ModelIR,
    op: OperatorIR,
    indices: Sequence[int],
) -> bool:
    bias_names = [_sequence_lstm_input_name(op, int(index)) for index in list(indices)]
    non_empty_bias_names = [name for name in bias_names if name != ""]
    if len(non_empty_bias_names) == 0:
        return True
    if len(non_empty_bias_names) != len(bias_names):
        return False
    return all(_tensor_has_constant_data(model_ir, name) for name in non_empty_bias_names)

def _sequence_lstm_index_spec(op: OperatorIR) -> Optional[Dict[str, Any]]:
    op_type = str(op.op_type)
    input_count = int(len(op.inputs))
    if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        if input_count >= 24:
            return {
                "required_const_indices": [1, 2, 3, 4, 5, 6, 7, 8],
                "unsupported_optional_indices": [9, 10, 11, 16, 17, 20, 21, 22, 23],
                "weight_input_indices": [1, 2, 3, 4],
                "recurrent_input_indices": [5, 6, 7, 8],
                "bias_indices": [12, 13, 14, 15],
                "state_indices": [18, 19],
            }
        if input_count == 15:
            return {
                "required_const_indices": [1, 2, 3, 4, 5, 6, 7, 8],
                "unsupported_optional_indices": [],
                "weight_input_indices": [1, 2, 3, 4],
                "recurrent_input_indices": [5, 6, 7, 8],
                "bias_indices": [9, 10, 11, 12],
                "state_indices": [13, 14],
            }
        return None
    if op_type == "BIDIRECTIONAL_SEQUENCE_LSTM":
        if input_count >= 48:
            return {
                "required_const_indices": [
                    1, 2, 3, 4, 5, 6, 7, 8,
                    18, 19, 20, 21, 22, 23, 24, 25,
                ],
                "unsupported_optional_indices": [9, 10, 11, 16, 17, 26, 27, 28, 33, 34, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                "fw_weight_input_indices": [1, 2, 3, 4],
                "fw_recurrent_input_indices": [5, 6, 7, 8],
                "fw_bias_indices": [12, 13, 14, 15],
                "bw_weight_input_indices": [18, 19, 20, 21],
                "bw_recurrent_input_indices": [22, 23, 24, 25],
                "bw_bias_indices": [29, 30, 31, 32],
                "state_indices": [35, 36, 37, 38],
            }
        if input_count == 29:
            return {
                "required_const_indices": [
                    1, 2, 3, 4, 5, 6, 7, 8,
                    13, 14, 15, 16, 17, 18, 19, 20,
                ],
                "unsupported_optional_indices": [],
                "fw_weight_input_indices": [1, 2, 3, 4],
                "fw_recurrent_input_indices": [5, 6, 7, 8],
                "fw_bias_indices": [9, 10, 11, 12],
                "bw_weight_input_indices": [13, 14, 15, 16],
                "bw_recurrent_input_indices": [17, 18, 19, 20],
                "bw_bias_indices": [21, 22, 23, 24],
                "state_indices": [25, 26, 27, 28],
            }
        return None
    return None

def _can_direct_codegen_sequence_lstm_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    if op_type not in {"UNIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_LSTM"}:
        return False
    index_spec = _sequence_lstm_index_spec(op)
    if index_spec is None:
        return False
    options = dict(op.options)
    if not bool(options.get("timeMajor", True)):
        return False
    if str(options.get("fusedActivationFunction", "TANH")).upper() != "TANH":
        return False
    if abs(float(options.get("cellClip", 0.0))) > 1e-12:
        return False
    if abs(float(options.get("projClip", 0.0))) > 1e-12:
        return False
    if len(op.outputs) != 1:
        return False

    required_const_indices = list(index_spec["required_const_indices"])
    unsupported_optional_indices = list(index_spec["unsupported_optional_indices"])
    if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        bias_indices = list(index_spec["bias_indices"])
    else:
        bias_indices = list(index_spec["fw_bias_indices"]) + list(index_spec["bw_bias_indices"])

    if any(not _tensor_has_constant_data(model_ir, _sequence_lstm_input_name(op, idx)) for idx in required_const_indices):
        return False
    if any(_sequence_lstm_input_name(op, idx) != "" for idx in unsupported_optional_indices):
        return False
    if not _sequence_lstm_bias_inputs_supported(model_ir, op, bias_indices):
        return False
    return True

def _can_direct_codegen_sequence_rnn_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    if str(op.op_type) != "UNIDIRECTIONAL_SEQUENCE_RNN":
        return False
    options = dict(op.options)
    if not bool(options.get("timeMajor", True)):
        return False
    if str(options.get("fusedActivationFunction", "TANH")).upper() not in {"TANH", "RELU"}:
        return False
    if len(op.outputs) != 1 or len(op.inputs) < 4:
        return False
    required_const_indices = [1, 2, 3]
    if any(
        not _tensor_has_constant_data(model_ir, _sequence_lstm_input_name(op, idx))
        for idx in required_const_indices
    ):
        return False
    weight_name = _sequence_lstm_input_name(op, 1)
    recurrent_name = _sequence_lstm_input_name(op, 2)
    bias_name = _sequence_lstm_input_name(op, 3)
    if weight_name == "" or recurrent_name == "" or bias_name == "":
        return False
    weight_tensor = model_ir.tensors.get(weight_name, None)
    recurrent_tensor = model_ir.tensors.get(recurrent_name, None)
    bias_tensor = model_ir.tensors.get(bias_name, None)
    if weight_tensor is None or recurrent_tensor is None or bias_tensor is None:
        return False
    weight_shape = [int(v) for v in list(weight_tensor.shape)]
    recurrent_shape = [int(v) for v in list(recurrent_tensor.shape)]
    bias_shape = [int(v) for v in list(bias_tensor.shape)]
    if len(weight_shape) != 2 or len(recurrent_shape) != 2 or len(bias_shape) != 1:
        return False
    hidden_size = int(weight_shape[0])
    return (
        hidden_size > 0
        and int(recurrent_shape[0]) == hidden_size
        and int(recurrent_shape[1]) == hidden_size
        and int(bias_shape[0]) == hidden_size
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
    boundary_shape_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
    public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    public_layout_bridge_tensor_names = model_ir.metadata.get("public_layout_bridge_tensor_names", [])
    if not isinstance(public_layout_bridge_tensor_names, list):
        public_layout_bridge_tensor_names = []
    public_tensor_names = {
        str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)
    }
    current_public_layouts: Dict[str, str] = {}
    tensors: Dict[str, Dict[str, Any]] = {}
    for name, tensor in model_ir.tensors.items():
        tensor_name = str(name)
        tensor_meta = _serializable_tensor_meta(tensor)
        if tensor_name in public_tensor_names:
            current_public_layouts[tensor_name] = str(tensor_meta["logical_layout"])
            boundary_shape = boundary_shape_map.get(tensor_name, None)
            if isinstance(boundary_shape, list) and len(boundary_shape) == len(tensor_meta["shape"]):
                tensor_meta["shape"] = [
                    max(1, int(v)) if int(v) >= 0 else 1
                    for v in list(boundary_shape)
                ]
                tensor_meta["shape_signature"] = [int(v) for v in list(boundary_shape)]
            public_layout = normalize_logical_layout(public_layout_map.get(tensor_name, None))
            if logical_layout_rank(public_layout) == len(tensor_meta["shape"]):
                tensor_meta["logical_layout"] = public_layout
        tensors[tensor_name] = tensor_meta
    for bridge_name_raw in public_layout_bridge_tensor_names:
        bridge_name = str(bridge_name_raw)
        if bridge_name == "" or bridge_name in tensors:
            continue
        base_name = bridge_name.removesuffix("_public_layout_bridge")
        base_tensor = model_ir.tensors.get(base_name, None)
        if base_tensor is None:
            continue
        tensor_meta = _serializable_tensor_meta(base_tensor)
        boundary_shape = boundary_shape_map.get(base_name, None)
        public_layout = normalize_logical_layout(public_layout_map.get(base_name, None))
        rank = len(list(tensor_meta["shape"]))
        internal_layout = (
            channel_last_logical_layout(rank)
            if is_channel_first_logical_layout(public_layout)
            else channel_first_logical_layout(rank)
            if is_channel_last_logical_layout(public_layout)
            else normalize_logical_layout(tensor_meta["logical_layout"])
        )
        if (
            isinstance(boundary_shape, list)
            and len(boundary_shape) == rank
            and public_layout != LOGICAL_LAYOUT_UNKNOWN
            and internal_layout != LOGICAL_LAYOUT_UNKNOWN
            and public_layout != internal_layout
        ):
            perm_to_internal = logical_layout_permutation(
                source_layout=public_layout,
                target_layout=internal_layout,
            )
            internal_shape = (
                None
                if perm_to_internal is None
                else _permute_shape([int(v) for v in list(boundary_shape)], perm_to_internal)
            )
            if internal_shape is not None:
                tensor_meta["shape"] = [max(1, int(v)) if int(v) >= 0 else 1 for v in list(internal_shape)]
                tensor_meta["shape_signature"] = [int(v) for v in list(internal_shape)]
        if logical_layout_rank(internal_layout) == rank:
            tensor_meta["logical_layout"] = internal_layout
        tensor_meta["has_data"] = False
        tensors[bridge_name] = tensor_meta
    return {
        "schema_version": 1,
        "name": str(model_ir.name),
        "description": str(model_ir.description),
        "inputs": [str(v) for v in model_ir.inputs],
        "outputs": [str(v) for v in model_ir.outputs],
        "tensors": tensors,
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
        "current_public_layouts": _serializable_value(current_public_layouts),
        "boundary_shape_signatures": _serializable_value(boundary_shape_map),
    }

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
    "FULLY_CONNECTED",
    "PRELU",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
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
    "GELU": "F.gelu({x})",
    "HARD_SWISH": "F.hardswish({x})",
    "IDENTITY": "{x}",
    "LEAKY_RELU": "F.leaky_relu({x}, negative_slope={alpha})",
    "LOG": "torch.log({x})",
    "LOGICAL_NOT": "torch.logical_not({x})",
    "LOGISTIC": "torch.sigmoid({x})",
    "NEG": "torch.neg({x})",
    "RELU": "torch.relu({x})",
    "RELU_0_TO_1": "torch.clamp({x}, min=0.0, max=1.0)",
    "RELU_N1_TO_1": "torch.clamp({x}, min=-1.0, max=1.0)",
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
    "ATAN2": "torch.atan2",
    "DIV": "torch.div",
    "EQUAL": "torch.eq",
    "FLOOR_MOD": "torch.remainder",
    "GREATER": "torch.gt",
    "GREATER_EQUAL": "torch.ge",
    "LESS": "torch.lt",
    "LOGICAL_AND": "torch.logical_and",
    "LOGICAL_OR": "torch.logical_or",
    "MAXIMUM": "torch.maximum",
    "MINIMUM": "torch.minimum",
    "MUL": "torch.mul",
    "NOT_EQUAL": "torch.ne",
    "POW": "torch.pow",
    "SUB": "torch.sub",
}

_DIRECT_CODEGEN_SUPPORTED_OP_TYPES: Set[str] = (
    set(_DIRECT_CODEGEN_MODULE_OP_TYPES)
    | set(_DIRECT_CODEGEN_UNARY_EXPRESSIONS.keys())
    | set(_DIRECT_CODEGEN_BINARY_FUNCTIONS.keys())
    | {
        "ARG_MAX",
        "ARG_MIN",
        "AVERAGE_POOL_2D",
        "BATCH_MATMUL",
        "CAST",
        "CONCATENATION",
        "CUMSUM",
        "DEPTH_TO_SPACE",
        "EXPAND_DIMS",
        "FILL",
        "GATHER",
        "GATHER_ND",
        "MAX_POOL_2D",
        "LOCAL_RESPONSE_NORMALIZATION",
        "MEAN",
        "MIRROR_PAD",
        "NON_MAX_SUPPRESSION_V4",
        "PACK",
        "PAD",
        "PADV2",
        "RANDOM_STANDARD_NORMAL",
        "RANGE",
        "REDUCE_ANY",
        "REDUCE_MAX",
        "REDUCE_MIN",
        "REDUCE_PROD",
        "REVERSE_V2",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SCATTER_ND",
        "SHAPE",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUM",
        "TOPK_V2",
        "TILE",
        "TRANSPOSE",
        "UNPACK",
        "SELECT",
        "SELECT_V2",
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

_PYTORCH_LOCAL_NAME_MAX_LENGTH = 40

_GENERATED_NAME_DROP_TOKENS = {
    "model",
    "readvariableop",
    "read",
    "variable",
    "resource",
}

_GENERATED_NAME_TOKEN_ALIASES = {
    "block": "b",
    "pool": "p",
    "conv": "cv",
    "conv2d": "cv",
    "depthwise": "dw",
    "depthwise1": "dw",
    "fusedbatchnorm": "bn",
    "fusedbatchnormv3": "bn",
    "batchnorm": "bn",
    "bn": "bn",
    "relu": "relu",
    "sigmoid": "sig",
    "add": "add",
    "mul": "mul",
    "mean": "mean",
    "transpose": "tr",
    "reshape": "rs",
    "input": "in",
    "output": "out",
    "channel": "ch",
    "first": "first",
    "nhwc": "nhwc",
}

_GENERATED_NAME_SUFFIX_PATTERNS: Sequence[Tuple[str, List[str]]] = (
    ("_input_nhwc__channel_first", ["in", "cf"]),
    ("_output_nhwc__channel_first", ["out", "cf"]),
    ("_input_nwc__channel_first", ["in", "cf"]),
    ("_output_nwc__channel_first", ["out", "cf"]),
    ("_input_ndhwc__channel_first", ["in", "cf"]),
    ("_output_ndhwc__channel_first", ["out", "cf"]),
    ("__channel_first", ["cf"]),
    ("_input_nhwc", ["in", "nhwc"]),
    ("_output_nhwc", ["out", "nhwc"]),
    ("_input_nwc", ["in", "nhwc"]),
    ("_output_nwc", ["out", "nhwc"]),
    ("_input_ndhwc", ["in", "nhwc"]),
    ("_output_ndhwc", ["out", "nhwc"]),
    ("_input", ["in"]),
    ("_output", ["out"]),
)

def _extract_generated_name_suffix_tokens(name: str) -> Tuple[str, List[str]]:
    lowered = str(name).lower()
    for raw_suffix, suffix_tokens in _GENERATED_NAME_SUFFIX_PATTERNS:
        if lowered.endswith(raw_suffix):
            return str(name)[: len(str(name)) - len(raw_suffix)], list(suffix_tokens)
    return str(name), []

def _split_generated_name_piece(piece: str) -> List[str]:
    compact_piece = re.sub(r"[^0-9A-Za-z]+", "", str(piece))
    if compact_piece == "":
        return []
    lowered_piece = compact_piece.lower()
    if lowered_piece in _GENERATED_NAME_DROP_TOKENS:
        return []
    for pattern, token_template in (
        (r"conv2d(\d+)", "cv{}"),
        (r"conv(\d+)", "cv{}"),
        (r"depthwise(\d+)", "dw{}"),
        (r"relu(\d+)", "relu{}"),
        (r"sigmoid(\d+)", "sig{}"),
        (r"add(\d+)", "add{}"),
        (r"mul(\d+)", "mul{}"),
        (r"mean(\d+)", "mean{}"),
        (r"reshape(\d+)", "rs{}"),
        (r"transpose(\d+)", "tr{}"),
        (r"pool(\d+)", "p{}"),
        (r"block(\d+)", "b{}"),
        (r"bn(\d+)", "bn{}"),
    ):
        matched = re.fullmatch(pattern, lowered_piece)
        if matched is not None:
            return [str(token_template).format(matched.group(1))]
    whole_alias = _GENERATED_NAME_TOKEN_ALIASES.get(lowered_piece, None)
    if whole_alias is not None:
        return [str(whole_alias)]
    split_tokens: List[str] = []
    for fragment in re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+", compact_piece):
        lowered_fragment = str(fragment).lower()
        if lowered_fragment in _GENERATED_NAME_DROP_TOKENS or lowered_fragment == "":
            continue
        split_tokens.append(
            str(_GENERATED_NAME_TOKEN_ALIASES.get(lowered_fragment, lowered_fragment))
        )
    return split_tokens if len(split_tokens) > 0 else [lowered_piece]

def _collapse_generated_name_tokens(tokens: Sequence[str]) -> List[str]:
    collapsed: List[str] = []
    for token in tokens:
        token_str = str(token)
        if token_str == "":
            continue
        if (
            token_str.isdigit()
            and len(collapsed) > 0
            and not collapsed[-1].isdigit()
            and not bool(re.search(r"\d$", collapsed[-1]))
        ):
            collapsed[-1] = f"{collapsed[-1]}{token_str}"
            continue
        if len(collapsed) > 0 and collapsed[-1] == token_str:
            continue
        collapsed.append(token_str)
    return collapsed

def _shorten_generated_python_identifier(
    name: str,
    *,
    prefix: str,
    max_length: int = _PYTORCH_LOCAL_NAME_MAX_LENGTH,
) -> str:
    stem_name, suffix_tokens = _extract_generated_name_suffix_tokens(str(name))
    base_tokens: List[str] = []
    for piece in re.split(r"_+", str(stem_name)):
        base_tokens.extend(_split_generated_name_piece(piece))
    base_tokens = _collapse_generated_name_tokens(base_tokens)
    candidate_tokens = base_tokens + list(suffix_tokens)
    candidate = _sanitize_python_identifier("_".join(candidate_tokens), prefix=prefix)
    if len(candidate) <= int(max_length):
        return candidate
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:4]
    core_tokens = list(base_tokens)
    leading_tokens = core_tokens[:3]
    trailing_tokens: List[str] = []
    for token in reversed(core_tokens):
        if token in leading_tokens or token in trailing_tokens:
            continue
        trailing_tokens.insert(0, token)
        if len(trailing_tokens) >= 2:
            break
    compressed_tokens = _collapse_generated_name_tokens(
        [*leading_tokens, *trailing_tokens, *suffix_tokens, digest]
    )
    candidate = _sanitize_python_identifier("_".join(compressed_tokens), prefix=prefix)
    trim_tokens = list(compressed_tokens[:-1])
    while len(candidate) > int(max_length) and len(trim_tokens) > 0:
        trim_index = max(
            range(len(trim_tokens)),
            key=lambda idx: (
                len(trim_tokens[idx])
                if trim_tokens[idx] not in {"in", "out", "cf", "nhwc"}
                else -1
            ),
        )
        if len(trim_tokens[trim_index]) > 3:
            trim_tokens[trim_index] = trim_tokens[trim_index][:-1]
        elif len(trim_tokens) > 1:
            del trim_tokens[trim_index]
        else:
            break
        candidate = _sanitize_python_identifier("_".join([*trim_tokens, digest]), prefix=prefix)
    if len(candidate) <= int(max_length):
        return candidate
    fallback = _sanitize_python_identifier(f"{prefix}_{digest}", prefix=prefix)
    return fallback[: int(max_length)]

def _import_generated_package_from_output(package_path: str) -> Any:
    package_dir = Path(package_path)
    package_name = re.sub(r"[^0-9A-Za-z_]", "_", package_dir.name).strip("_")
    if package_name == "":
        package_name = "generated_pytorch_package"
    module_name = f"_onnx2tf_generated_{package_name}"
    stale_module_names = [
        existing_name
        for existing_name in list(sys.modules.keys())
        if existing_name == module_name or existing_name.startswith(f"{module_name}.")
    ]
    for existing_name in stale_module_names:
        sys.modules.pop(existing_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ModelIRPyTorchExportError(
            f"Could not import generated PyTorch package for state_dict export. path={package_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _prepare_exported_state_tensor(source: np.ndarray, target: Any) -> Any:
    import torch

    source_tensor = torch.as_tensor(np.asarray(source))
    target_tensor = torch.as_tensor(target)
    candidate = source_tensor.to(dtype=target_tensor.dtype)
    if list(candidate.shape) == list(target_tensor.shape):
        return candidate.detach().cpu().clone()
    if int(candidate.numel()) == int(target_tensor.numel()):
        reshaped = candidate.reshape(target_tensor.shape)
        if list(reshaped.shape) == list(target_tensor.shape):
            return reshaped.detach().cpu().clone()
    perm = _perm_cl_to_cf(candidate.ndim)
    if perm is not None:
        permuted = candidate.permute(*perm).contiguous()
        if list(permuted.shape) == list(target_tensor.shape):
            return permuted.detach().cpu().clone()
    if candidate.ndim <= 5:
        import itertools

        for generic_perm in itertools.permutations(range(candidate.ndim)):
            if list(generic_perm) == list(range(candidate.ndim)):
                continue
            permuted = candidate.permute(*generic_perm).contiguous()
            if list(permuted.shape) == list(target_tensor.shape):
                return permuted.detach().cpu().clone()
    raise ModelIRPyTorchExportError(
        "Native PyTorch state_dict export could not align a tensor to the generated module shape. "
        f"source_shape={list(candidate.shape)} target_shape={list(target_tensor.shape)}"
    )

def _build_native_generated_state_dict(
    *,
    package_path: str,
    model_ir: ModelIR,
    load_specs: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    package_module = _import_generated_package_from_output(package_path)
    model = package_module.Model(load_weights=False)
    exported_state_dict = {
        str(key): value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    expected_keys = {str(key) for key in exported_state_dict.keys()}
    mapped_keys = {str(attr_path) for attr_path, _ in load_specs}
    if expected_keys != mapped_keys:
        raise ModelIRPyTorchExportError(
            "Native PyTorch state_dict export could not reconcile generated state_dict keys. "
            f"expected_keys={sorted(expected_keys)} mapped_keys={sorted(mapped_keys)}"
        )
    for attr_path, tensor_name in load_specs:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or not isinstance(tensor.data, np.ndarray):
            raise ModelIRPyTorchExportError(
                "Native PyTorch state_dict export requires concrete tensor data for every generated state entry. "
                f"tensor={tensor_name}"
            )
        exported_state_dict[str(attr_path)] = _prepare_exported_state_tensor(
            np.asarray(tensor.data),
            exported_state_dict[str(attr_path)],
        )
    return exported_state_dict

def _build_tensor_var_name_map(model_ir: ModelIR) -> Dict[str, str]:
    used_names: Set[str] = set()
    mapping: Dict[str, str] = {}

    def _canonical_tensor_var_source_name(tensor_name: str) -> str:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        base_name = str(tensor_name)
        if tensor is not None:
            layout = normalize_logical_layout(tensor.logical_layout)
            if not is_channel_last_logical_layout(layout):
                base_name = re.sub(
                    r"_(?:nhwc|nwc|ndhwc)$",
                    "",
                    base_name,
                    flags=re.IGNORECASE,
                )
        return base_name

    for tensor_name in (
        list(model_ir.inputs)
        + [str(out) for op in model_ir.operators for out in op.outputs]
        + list(model_ir.outputs)
    ):
        if str(tensor_name) in mapping:
            continue
        base = _shorten_generated_python_identifier(
            _canonical_tensor_var_source_name(str(tensor_name)),
            prefix="t",
        )
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

def _build_model_ir_producer_consumer_index(
    model_ir: ModelIR,
) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    producers: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_index, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_index)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_index))
    return producers, consumers

def _is_small_inline_constant_tensor(tensor: TensorIR) -> bool:
    if tensor.data is None:
        return False
    arr = np.asarray(tensor.data)
    if arr.size > 32:
        return False
    if arr.ndim > 2:
        return False
    return str(tensor.dtype).upper() in {
        "BOOL",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "FLOAT16",
        "FLOAT32",
        "FLOAT64",
    }

def _python_literal_for_constant_tensor(tensor: TensorIR) -> Optional[str]:
    if not _is_small_inline_constant_tensor(tensor):
        return None
    def _python_literal_value(value: Any) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            value = float(value)
            if math.isnan(value):
                return "float('nan')"
            if math.isinf(value):
                return "float('inf')" if value > 0.0 else "float('-inf')"
            return repr(value)
        if isinstance(value, list):
            return "[" + ", ".join(_python_literal_value(item) for item in value) + "]"
        return repr(value)

    arr = np.asarray(tensor.data)
    if arr.ndim == 0:
        return _python_literal_value(arr.reshape(-1)[0].item())
    return _python_literal_value(arr.tolist())

def _torch_pad_literal_for_constant_tensor(
    tensor: Optional[TensorIR],
    *,
    axis_permutation: Optional[Sequence[int]] = None,
) -> Optional[str]:
    if tensor is None or tensor.data is None:
        return None
    pads = np.asarray(tensor.data).astype(np.int64).reshape(-1, 2).tolist()
    if axis_permutation is not None:
        perm = [int(v) for v in list(axis_permutation)]
        if len(perm) == len(pads):
            pads = [pads[idx] for idx in perm]
    torch_pad: List[int] = []
    for before, after in reversed(pads):
        torch_pad.extend([int(before), int(after)])
    while len(torch_pad) >= 2 and int(torch_pad[-2]) == 0 and int(torch_pad[-1]) == 0:
        torch_pad = torch_pad[:-2]
    return repr(torch_pad)

def _scalar_literal_for_constant_tensor(tensor: Optional[TensorIR]) -> Optional[str]:
    if tensor is None or tensor.data is None:
        return None
    flat = np.asarray(tensor.data).reshape(-1)
    if int(flat.size) != 1:
        return None
    value = flat[0].item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        value = float(value)
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0.0 else "float('-inf')"
        return repr(value)
    return repr(value)

def _constant_int_list(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None or tensor.data is None:
        return None
    arr = np.asarray(tensor.data)
    if arr.size == 0:
        return []
    if not np.issubdtype(arr.dtype, np.integer):
        return None
    return [int(v) for v in arr.reshape(-1).tolist()]

def _torch_dtype_literal(dtype_name: str) -> str:
    mapping = {
        "BOOL": "torch.bool",
        "INT8": "torch.int8",
        "INT16": "torch.int16",
        "INT32": "torch.int32",
        "INT64": "torch.int64",
        "UINT8": "torch.uint8",
        "FLOAT16": "torch.float16",
        "FLOAT32": "torch.float32",
        "FLOAT64": "torch.float64",
    }
    key = str(dtype_name).upper()
    if key not in mapping:
        raise ModelIRPyTorchExportError(
            f"Unsupported dtype for native PyTorch-like model.py codegen: {dtype_name}"
        )
    return str(mapping[key])

def _conv_block_activation_config(op: OperatorIR) -> Tuple[str, Optional[float]]:
    op_type = str(op.op_type)
    if op_type == "LEAKY_RELU":
        return ("leaky_relu", float(op.options.get("alpha", 0.2)))
    if op_type == "RELU":
        return ("relu", None)
    if op_type == "RELU6":
        return ("relu6", None)
    if op_type == "RELU_N1_TO_1":
        return ("relu_n1_to_1", None)
    if op_type == "RELU_0_TO_1":
        return ("relu_0_to_1", None)
    if op_type == "TANH":
        return ("tanh", None)
    if op_type == "LOGISTIC":
        return ("sigmoid", None)
    return ("none", None)

def _conv_block_activation_config_from_fused_name(
    fused_name: str,
    *,
    alpha: Optional[float] = None,
) -> Tuple[str, Optional[float]]:
    key = str(fused_name).upper()
    if key == "LEAKY_RELU":
        return ("leaky_relu", float(0.2 if alpha is None else alpha))
    if key == "RELU":
        return ("relu", None)
    if key == "RELU6":
        return ("relu6", None)
    if key == "RELU_N1_TO_1":
        return ("relu_n1_to_1", None)
    if key == "RELU_0_TO_1":
        return ("relu_0_to_1", None)
    if key == "TANH":
        return ("tanh", None)
    if key == "LOGISTIC":
        return ("sigmoid", None)
    return ("none", None)

def _reshape_special_layout_plan(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    input_layout: Optional[str],
    output_layout: Optional[str],
) -> Optional[Dict[str, Any]]:
    if input_shape is None or output_shape is None:
        return None
    src = [int(v) for v in list(input_shape)]
    dst = [int(v) for v in list(output_shape)]
    in_layout = str(input_layout or "").upper()
    out_layout = str(output_layout or "").upper()
    if (
        in_layout == "NCHW"
        and len(src) == 4
        and len(dst) == 3
        and int(src[0]) == 1
        and int(src[2]) == int(dst[0])
        and int(src[3]) == int(dst[1])
        and int(src[1]) == int(dst[2])
    ):
        return {
            "pre_perm": [0, 2, 3, 1],
            "reshape_shape": list(dst),
            "post_perm": None,
        }
    if (
        len(src) == 3
        and len(dst) == 4
        and int(dst[0]) == 1
        and int(src[0]) == int(dst[2])
        and int(src[1]) == int(dst[3])
        and int(src[2]) == int(dst[1])
        and out_layout == "NCHW"
    ):
        return {
            "pre_perm": None,
            "reshape_shape": [1, int(src[0]), int(src[1]), int(src[2])],
            "post_perm": [0, 3, 1, 2],
        }
    if (
        len(src) == 4
        and len(dst) == 4
        and int(src[0]) == int(dst[0])
        and any(int(v) == 1 for v in src[1:])
    ):
        if [int(src[0]), int(src[3]), int(src[1]), int(src[2])] == dst:
            return {
                "pre_perm": [0, 3, 1, 2],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
        if [int(src[0]), int(src[2]), int(src[3]), int(src[1])] == dst:
            return {
                "pre_perm": [0, 2, 3, 1],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
    if (
        in_layout == "NCHW"
        and out_layout == "NCDHW"
        and len(src) == 4
        and len(dst) == 5
        and int(src[0]) == int(dst[0])
        and int(src[1]) == 1
        and int(dst[2]) == 1
        and int(dst[3]) == 1
        and int(src[2]) == int(dst[4])
        and int(src[3]) == int(dst[1])
    ):
        return {
            "pre_perm": [0, 3, 1, 2],
            "reshape_shape": list(dst),
            "post_perm": None,
        }
    if (
        in_layout == "NCHW"
        and len(src) == 4
        and len(dst) >= 5
        and int(src[0]) == int(dst[0])
        and int(src[2]) == int(dst[1])
        and int(src[3]) == int(dst[2])
    ):
        trailing_product = 1
        for dim in dst[3:]:
            trailing_product *= int(dim)
        if int(src[1]) == int(trailing_product):
            return {
                "pre_perm": [0, 2, 3, 1],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
    return None

def _reshape_is_plain_singleton_axis_drop(
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
) -> bool:
    if input_shape is None or output_shape is None:
        return False
    src = [int(v) for v in list(input_shape)]
    dst = [int(v) for v in list(output_shape)]
    if len(src) != len(dst) + 1:
        return False
    singleton_axes = [axis for axis, dim in enumerate(src) if int(dim) == 1]
    if len(singleton_axes) != 1:
        return False
    axis = int(singleton_axes[0])
    return src[:axis] + src[axis + 1:] == dst

def _direct_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    size_values: Sequence[int],
    input_rank: int,
    input_shape: Optional[Sequence[int]] = None,
) -> Optional[str]:
    if len(begin_values) != int(input_rank) or len(size_values) != int(input_rank):
        return None
    direct_x_expr = f"{x_expr}.reshape(-1)" if int(input_rank) == 1 else x_expr
    parts: List[str] = []
    for axis, (start, length) in enumerate(zip(begin_values, size_values)):
        dim_size: Optional[int] = None
        if input_shape is not None and axis < len(input_shape):
            try:
                dim_size = int(input_shape[axis])
            except Exception:
                dim_size = None
        resolved_start = int(start)
        if int(length) < 0:
            resolved_stop: Optional[int] = None
        else:
            resolved_stop = resolved_start + int(length)
            if dim_size is not None and int(dim_size) > 0:
                resolved_stop = min(int(resolved_stop), int(dim_size))
        if resolved_start == 0 and resolved_stop is None:
            parts.append(":")
        else:
            start_str = "" if resolved_start == 0 else str(resolved_start)
            stop_str = "" if resolved_stop is None else str(int(resolved_stop))
            parts.append(f"{start_str}:{stop_str}")
    return f"{direct_x_expr}[{', '.join(parts)}]"

def _direct_strided_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    end_values: Sequence[int],
    stride_values: Sequence[int],
    begin_mask: int,
    end_mask: int,
    input_rank: int,
) -> Optional[str]:
    if (
        len(begin_values) != int(input_rank)
        or len(end_values) != int(input_rank)
        or len(stride_values) != int(input_rank)
    ):
        return None
    direct_x_expr = f"{x_expr}.reshape(-1)" if int(input_rank) == 1 else x_expr
    parts: List[str] = []
    for axis, (start, stop, step) in enumerate(zip(begin_values, end_values, stride_values)):
        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)
        resolved_stop = None if ((int(end_mask) >> axis) & 1) else int(stop)
        if resolved_stop is not None and int(resolved_stop) >= 2147483647:
            resolved_stop = None
        resolved_step = int(step)
        if resolved_step == 0:
            return None
        if resolved_start is None and resolved_stop is None and resolved_step == 1:
            parts.append(":")
            continue
        start_str = "" if resolved_start is None else str(int(resolved_start))
        stop_str = "" if resolved_stop is None else str(int(resolved_stop))
        if resolved_step == 1:
            parts.append(f"{start_str}:{stop_str}")
        else:
            parts.append(f"{start_str}:{stop_str}:{resolved_step}")
    return f"{direct_x_expr}[{', '.join(parts)}]"

def _direct_symbolic_strided_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    stride_values: Sequence[int],
    begin_mask: int,
    end_mask: int,
    input_rank: int,
    end_list_expr: Optional[str] = None,
    end_scalar_expr: Optional[str] = None,
) -> Optional[str]:
    if len(begin_values) != int(input_rank) or len(stride_values) != int(input_rank):
        return None
    if end_list_expr is None and end_scalar_expr is None:
        return None
    direct_x_expr = f"{x_expr}.reshape(-1)" if int(input_rank) == 1 else x_expr
    parts: List[str] = []
    for axis, (start, step) in enumerate(zip(begin_values, stride_values)):
        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)
        if ((int(end_mask) >> axis) & 1):
            resolved_stop_expr = None
        elif int(input_rank) == 1 and end_scalar_expr is not None:
            resolved_stop_expr = str(end_scalar_expr)
        elif end_list_expr is not None:
            resolved_stop_expr = f"({end_list_expr})[{int(axis)}]"
        else:
            return None
        resolved_step = int(step)
        if resolved_step == 0:
            return None
        start_str = "" if resolved_start is None or int(resolved_start) == 0 else str(int(resolved_start))
        stop_str = "" if resolved_stop_expr is None else str(resolved_stop_expr)
        if resolved_step == 1:
            parts.append(f"{start_str}:{stop_str}")
        else:
            parts.append(f"{start_str}:{stop_str}:{resolved_step}")
    return f"{direct_x_expr}[{', '.join(parts)}]"

def _direct_gather_expr(
    *,
    params_expr: str,
    indices_values: Sequence[int],
    indices_shape: Optional[Sequence[int]],
    axis: int,
    batch_dims: int,
    input_rank: int,
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    if len(indices_values) == 0:
        return None
    normalized_indices_shape = (
        [int(v) for v in list(indices_shape)]
        if indices_shape is not None and len(list(indices_shape)) > 0
        else [int(len(indices_values))]
    )
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    if indices_shape is not None and len(list(indices_shape)) == 0:
        literal = int(indices_values[0])
        return (
            f"torch.index_select({params_expr}, {resolved_axis}, "
            f"torch.as_tensor([{literal}], dtype=torch.int64, device={params_expr}.device))"
            f".squeeze({resolved_axis})"
        )
    if len(normalized_indices_shape) > 1:
        literal = repr([int(v) for v in indices_values])
        return (
            f"_reshape_gather_output("
            f"torch.index_select({params_expr}, {resolved_axis}, "
            f"torch.as_tensor({literal}, dtype=torch.int64, device={params_expr}.device)), "
            f"{params_expr}, {repr(normalized_indices_shape)}, axis={resolved_axis})"
        )
    if int(input_rank) == 1 and int(resolved_axis) == 0:
        return f"{params_expr}.reshape(-1)[{repr([int(v) for v in indices_values])}]"
    parts = [":" for _ in range(int(input_rank))]
    parts[resolved_axis] = repr([int(v) for v in indices_values])
    return f"{params_expr}[{', '.join(parts)}]"

def _direct_dynamic_gather_expr(
    *,
    params_expr: str,
    indices_expr: str,
    axis: int,
    batch_dims: int,
    input_rank: int,
    indices_name: str,
    indices_shape: Optional[Sequence[int]] = None,
    indices_shape_signature: Optional[Sequence[int]] = None,
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    if str(indices_name).endswith("_crd_to_dcr_indices"):
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    flat_indices_expr = f"{indices_expr}.to(dtype=torch.int64).reshape(-1)"
    normalized_indices_shape = (
        [int(v) for v in list(indices_shape)]
        if indices_shape is not None
        else None
    )
    normalized_indices_shape_signature = (
        [int(v) for v in list(indices_shape_signature)]
        if indices_shape_signature is not None
        else None
    )
    indices_shape_expr = (
        repr(normalized_indices_shape)
        if (
            normalized_indices_shape is not None
            and all(int(v) > 0 for v in normalized_indices_shape)
            and (
                normalized_indices_shape_signature is None
                or all(int(v) > 0 for v in normalized_indices_shape_signature)
            )
        )
        else f"_shape_tensor({indices_expr}, dtype=torch.int64, device={indices_expr}.device)"
    )
    reshaped_expr = (
        f"_reshape_gather_output("
        f"torch.index_select({params_expr}, {resolved_axis}, {flat_indices_expr}), "
        f"{params_expr}, {indices_shape_expr}, axis={resolved_axis})"
    )
    return reshaped_expr

def _is_suffix_flatten_gather_reshape(
    gather_output_shape: Optional[Sequence[int]],
    reshape_output_shape: Optional[Sequence[int]],
) -> bool:
    if gather_output_shape is None or reshape_output_shape is None:
        return False
    gather_shape = [int(v) for v in list(gather_output_shape)]
    reshape_shape = [int(v) for v in list(reshape_output_shape)]
    if len(gather_shape) < 2 or len(reshape_shape) < 2 or len(reshape_shape) >= len(gather_shape):
        return False
    prefix_len = len(reshape_shape) - 1
    if prefix_len <= 0:
        return False
    for gather_dim, reshape_dim in zip(gather_shape[:prefix_len], reshape_shape[:prefix_len]):
        if int(gather_dim) == int(reshape_dim):
            continue
        if int(gather_dim) <= 0 or int(reshape_dim) <= 0:
            continue
        return False
    flattened_dims = gather_shape[prefix_len:]
    if len(flattened_dims) < 2 or any(int(dim) <= 0 for dim in flattened_dims):
        return False
    expected_flattened = 1
    for dim in flattened_dims:
        expected_flattened *= int(dim)
    reshape_last_dim = int(reshape_shape[-1])
    if reshape_last_dim <= 0:
        return False
    return int(expected_flattened) == int(reshape_last_dim)

def _direct_gather_reshape_expr(
    *,
    params_expr: str,
    indices_expr: str,
    indices_values: Optional[Sequence[int]],
    indices_shape: Optional[Sequence[int]],
    indices_shape_signature: Optional[Sequence[int]],
    axis: int,
    batch_dims: int,
    input_rank: int,
    indices_name: str,
    final_shape_values: Optional[Sequence[int]],
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    flat_indices_expr: Optional[str] = None
    if indices_values is not None:
        flat_indices_expr = (
            f"torch.as_tensor({repr([int(v) for v in list(indices_values)])}, "
            f"dtype=torch.int64, device={params_expr}.device)"
        )
    else:
        if str(indices_name).endswith("_crd_to_dcr_indices"):
            return None
        normalized_indices_shape = (
            [int(v) for v in list(indices_shape)]
            if indices_shape is not None
            else None
        )
        normalized_indices_shape_signature = (
            [int(v) for v in list(indices_shape_signature)]
            if indices_shape_signature is not None
            else None
        )
        if (
            normalized_indices_shape is None
            or len(normalized_indices_shape) <= 1
            or not all(int(v) > 0 for v in normalized_indices_shape)
            or (
                normalized_indices_shape_signature is not None
                and not all(int(v) > 0 for v in normalized_indices_shape_signature)
            )
        ):
            return None
        flat_indices_expr = f"{indices_expr}.to(dtype=torch.int64).reshape(-1)"
    if flat_indices_expr is None:
        return None
    if final_shape_values is None:
        return None
    resolved_shape_values = [int(v) for v in list(final_shape_values)]
    if len(resolved_shape_values) == 0:
        return None
    if all(int(v) > 0 for v in resolved_shape_values):
        final_shape_expr = repr(resolved_shape_values)
    elif (
        int(resolved_shape_values[0]) <= 0
        and all(int(v) > 0 for v in resolved_shape_values[1:])
    ):
        final_shape_expr = (
            f"[int({params_expr}.shape[0]), "
            + ", ".join(str(int(v)) for v in resolved_shape_values[1:])
            + "]"
        )
    else:
        return None
    return (
        f"torch.reshape("
        f"torch.index_select({params_expr}, {resolved_axis}, {flat_indices_expr}), "
        f"{final_shape_expr})"
    )

def _should_elide_crd_to_dcr_gather_for_depth_to_space(
    *,
    model_ir: ModelIR,
    params_name: str,
    indices_name: str,
    output_name: str,
    axis: int,
    batch_dims: int,
) -> bool:
    if int(batch_dims) != 0 or not str(indices_name).endswith("_crd_to_dcr_indices"):
        return False
    input_tensor = model_ir.tensors.get(str(params_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if input_tensor is None or output_tensor is None:
        return False
    input_rank = len(list(input_tensor.shape))
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis != 1:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if not (
        is_channel_first_logical_layout(input_layout)
        or is_channel_first_logical_layout(output_layout)
    ):
        return False
    consumer_op_types = [
        str(consumer.op_type)
        for consumer in model_ir.operators
        if str(output_name) in {str(v) for v in consumer.inputs}
    ]
    return len(consumer_op_types) > 0 and all(
        op_type == "DEPTH_TO_SPACE" for op_type in consumer_op_types
    )

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

def _write_generated_package_common_files(
    output_folder_path: str,
    *,
    runtime_source: Optional[str] = None,
) -> None:
    package_dir = Path(output_folder_path)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "from .model import Model, load_model\n",
        encoding="utf-8",
    )
    if runtime_source is None:
        runtime_source = (
            "# pyright: reportArgumentType=false, reportCallIssue=false\n"
            "from onnx2tf.tflite_builder.pytorch_package_runtime import load_generated_model_package\n"
        )
    (package_dir / "runtime.py").write_text(
        runtime_source,
        encoding="utf-8",
    )

def _write_wrapper_model_file(output_folder_path: str) -> None:
    package_dir = Path(output_folder_path)
    (package_dir / "model.py").write_text(
        "# pyright: reportArgumentType=false, reportCallIssue=false\n"
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

def _build_native_runtime_source(helper_source: str) -> str:
    runtime_source = (
        "# pyright: reportArgumentType=false, reportCallIssue=false\n"
        "from pathlib import Path\n"
        "import re\n"
        "from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple\n\n"
        "import numpy as np\n"
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        f"{helper_source}"
        "def _resolve_model_attribute(model: torch.nn.Module, attr_path: str) -> Any:\n"
        "    value: Any = model\n"
        "    for part in str(attr_path).split('.'):\n"
        "        value = getattr(value, part)\n"
        "    return value\n\n"
        "def resolve_model_tensor(model: torch.nn.Module, attr_name: str) -> torch.Tensor:\n"
        "    value = _resolve_model_attribute(model, attr_name)\n"
        "    if not isinstance(value, torch.Tensor):\n"
        "        raise RuntimeError(f'Generated model attribute is not a tensor: {attr_name}')\n"
        "    return value\n\n"
        "def load_generated_weights(\n"
        "    *,\n"
        "    model: torch.nn.Module,\n"
        "    package_dir: Path,\n"
        "    device: Optional[str],\n"
        ") -> None:\n"
        "    raw_state_dict = torch.load(package_dir / 'state_dict.pth', map_location=device or 'cpu')\n"
        "    model.load_state_dict(raw_state_dict, strict=True)\n"
        "    if device is not None:\n"
        "        model.to(device)\n"
    )
    runtime_source = runtime_source.replace("Optional[Sequence[int]]", "Optional[List[int]]")
    runtime_source = runtime_source.replace("Sequence[int]", "List[int]")
    runtime_source = runtime_source.replace("Sequence[str]", "List[str]")
    runtime_source = runtime_source.replace("Sequence[torch.Tensor]", "List[torch.Tensor]")
    runtime_source = runtime_source.replace("Sequence[Any]", "List[Any]")
    return runtime_source

def _direct_codegen_module_attr_base(op_type: str) -> str:
    names = {
        "CONV_2D": "conv2d",
        "DEPTHWISE_CONV_2D": "depthwise_conv2d",
        "TRANSPOSE_CONV": "conv_transpose2d",
        "CONV_3D": "conv3d",
        "CONV_3D_TRANSPOSE": "conv_transpose3d",
        "FULLY_CONNECTED": "linear",
        "PRELU": "prelu",
        "UNIDIRECTIONAL_SEQUENCE_RNN": "sequence_rnn",
        "UNIDIRECTIONAL_SEQUENCE_LSTM": "sequence_lstm",
        "BIDIRECTIONAL_SEQUENCE_LSTM": "bidirectional_sequence_lstm",
    }
    return str(names.get(str(op_type), str(op_type).lower()))

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
    metadata["execution_backend"] = "runtime_wrapper"
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
) -> List[Tuple[str, str]]:
    package_dir = Path(output_folder_path)
    _ensure_direct_codegen_supported(model_ir)
    producer_index, consumer_index = _build_model_ir_producer_consumer_index(model_ir)
    return _write_native_model_file_impl(
        _NativeModelFileWriterContext(
            output_folder_path,
            model_ir,
            metadata,
            tensor_storage_name_map,
            package_dir,
            _collect_feature_last_sequence_tensor_names(model_ir),
            _build_tensor_var_name_map(model_ir),
            producer_index,
            consumer_index,
        )
    )

def _write_native_model_file_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_impl(context)

def _write_native_model_file_codegen_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_impl(context)

def _write_native_model_file_codegen_core_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_body_impl(context)

def _write_native_model_file_codegen_core_body_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_body_main_impl(context)

def _write_native_model_file_codegen_core_body_main_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_body_main_inner_impl(context)

def _write_native_model_file_codegen_core_body_main_inner_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    state = _prepare_native_codegen_state(context)
    bindings = _build_native_codegen_bindings(state)
    _build_native_constant_aliases(state, bindings)
    _emit_native_forward_lines(state, bindings)
    return _finalize_native_codegen(state, bindings)

def _write_native_model_file_codegen_core_body_main_inner_legacy_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    output_folder_path = context.output_folder_path
    model_ir = context.model_ir
    metadata = context.metadata
    tensor_storage_name_map = context.tensor_storage_name_map
    package_dir = context.package_dir
    _ensure_direct_codegen_supported(model_ir)
    preserve_channel_last_tensor_names = context.preserve_channel_last_tensor_names
    tensor_var_names = context.tensor_var_names
    producer_index = context.producer_index
    consumer_index = context.consumer_index
    module_param_tensor_names: Set[str] = set()
    submodule_state_tensor_names: Set[str] = set()
    module_init_lines = context.module_init_lines
    load_specs = context.load_specs
    op_module_attr_names: Dict[int, str] = {}
    fused_module_specs: Dict[int, Dict[str, Any]] = {}
    affine_layer_norm_specs: Dict[int, Dict[str, Any]] = {}
    nms_method_specs: List[Dict[str, Any]] = []
    module_attr_counts: Dict[str, int] = {}
    inlined_constant_tensor_names: Set[str] = set()
    skipped_op_indices: Set[int] = set()
    conv_module_pad_specs: Dict[int, Optional[List[int]]] = {}
    tensor_expr_aliases: Dict[str, str] = {}
    channel_first_tensor_expr_aliases: Dict[str, str] = {}
    synthetic_local_var_names: Dict[str, str] = {}
    used_local_var_names: Set[str] = set(tensor_var_names.values())
    synthetic_tensor_serial_ref: List[int] = [0]
    public_input_names: Set[str] = {str(name) for name in list(model_ir.inputs)}
    public_layout_bridge_tensor_names: Set[str] = {
        str(name)
        for name in list(model_ir.metadata.get("public_layout_bridge_tensor_names", []))
        if str(name) != ""
    }

    _fold_single_consumer_public_input_bridge = lambda *, tensor_name, downstream_permute: _fold_single_consumer_public_input_bridge_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        consumer_index=consumer_index,
        public_layout_bridge_tensor_names=public_layout_bridge_tensor_names,
        public_input_names=public_input_names,
        tensor_name=tensor_name,
        downstream_permute=downstream_permute,
    )
    _match_single_consumer_layout_bridge_transpose = lambda *, tensor_name, required_output_layout=None: _match_single_consumer_layout_bridge_transpose_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        tensor_name=tensor_name,
        required_output_layout=required_output_layout,
    )
    _target_shape_values = lambda tensor_name: _target_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _target_shape_literal = lambda tensor_name: _target_shape_literal_for_model_ir(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _resize_target_shape_literal = lambda output_name, input_name: _resize_target_shape_literal_for_model_ir(
        model_ir=model_ir,
        output_name=output_name,
        input_name=input_name,
    )
    _tensor_shape_list = lambda tensor_name: _tensor_shape_list_for_model_ir(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _rank4_channel_first_shape_for_tensor = lambda tensor_name: _rank4_channel_first_shape_for_tensor_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        tensor_name=tensor_name,
    )
    _channel_first_shape_for_tensor = lambda tensor_name: _channel_first_shape_for_tensor_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        tensor_name=tensor_name,
    )

    _channel_first_concat_input_expr = lambda tensor_name: _channel_first_concat_input_expr_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        tensor_name=tensor_name,
        tensor_expr_fn=_tensor_expr,
    )
    _can_fold_channel_last_alias_slice_consumer = lambda op, *, expected_input_name: _can_fold_channel_last_alias_slice_consumer_for_codegen(
        model_ir=model_ir,
        op=op,
        expected_input_name=expected_input_name,
    )
    _resolve_concat_axis_for_channel_first = lambda op: _resolve_concat_axis_for_channel_first_for_codegen(
        model_ir=model_ir,
        op=op,
        channel_first_shape_for_tensor_fn=_channel_first_shape_for_tensor,
        tensor_shape_list_fn=_tensor_shape_list,
    )
    _can_keep_channel_first_slice_output = lambda output_name: _can_keep_channel_first_slice_output_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        output_name=output_name,
        resolve_concat_axis_for_channel_first_fn=_resolve_concat_axis_for_channel_first,
    )

    _reshape_codegen_is_plain_data_only = lambda op: _reshape_codegen_is_plain_data_only_for_codegen(
        model_ir=model_ir,
        op=op,
        infer_effective_rank4_runtime_layout_fn=_infer_effective_rank4_runtime_layout,
        reshape_preserves_channel_last_sequence_fn=_reshape_preserves_channel_last_sequence,
        reshape_prefers_feature_last_for_adjx_batch_matmul_fn=_reshape_prefers_feature_last_for_adjx_batch_matmul,
    )
    _tensor_exact_static_shape_list = lambda tensor_name: _tensor_exact_static_shape_list_for_model_ir(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _static_sequence_length = lambda tensor_name: _static_sequence_length_for_model_ir(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _is_identity_nms_postprocess_gather = lambda params_name, indices_name: _is_identity_nms_postprocess_gather_for_codegen(
        model_ir=model_ir,
        tensor_expr_aliases=tensor_expr_aliases,
        producer_index=producer_index,
        scalar_literal_expr_fn=_scalar_literal_expr,
        params_name=params_name,
        indices_name=indices_name,
    )
    _range_only_feeds_identity_nms_postprocess_gathers = lambda output_name: _range_only_feeds_identity_nms_postprocess_gathers_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        is_identity_nms_postprocess_gather_fn=_is_identity_nms_postprocess_gather,
        output_name=output_name,
    )
    _reshape_preserves_channel_last_sequence = lambda input_shape, output_shape, input_layout: _reshape_preserves_channel_last_sequence_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        input_layout=input_layout,
    )
    _reshape_prefers_feature_last_for_adjx_batch_matmul = lambda input_tensor_name, output_name: _reshape_prefers_feature_last_for_adjx_batch_matmul_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        input_tensor_name=input_tensor_name,
        output_name=output_name,
    )
    _matmul_broadcast_shape = lambda lhs_batch, rhs_batch: _matmul_broadcast_shape_for_codegen(
        lhs_batch=lhs_batch,
        rhs_batch=rhs_batch,
    )
    _infer_batch_matmul_shape = lambda lhs_shape, rhs_shape, *, adj_x, adj_y: _infer_batch_matmul_shape_for_codegen(
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        adj_x=adj_x,
        adj_y=adj_y,
    )
    _infer_reduction_shape = lambda input_shape, axes, *, keepdims: _infer_reduction_shape_for_codegen(
        input_shape=input_shape,
        axes=axes,
        keepdims=keepdims,
    )
    _infer_gather_nd_shape = lambda params_shape, indices_tensor_name: _infer_gather_nd_shape_for_codegen(
        model_ir=model_ir,
        params_shape=params_shape,
        indices_tensor_name=indices_tensor_name,
    )
    _emit_maybe_aligned_expr = lambda *, output_name, expr, inferred_shape: _emit_maybe_aligned_expr_for_codegen(
        runtime_imports=runtime_imports,
        output_name=output_name,
        expr=expr,
        inferred_shape=inferred_shape,
        tensor_shape_list_fn=_tensor_shape_list,
        target_shape_literal_fn=_target_shape_literal,
    )
    _emit_module_output_expr = lambda *, output_name, expr, raw_output_layout: _emit_module_output_expr_for_codegen(
        model_ir=model_ir,
        runtime_imports=runtime_imports,
        output_name=output_name,
        expr=expr,
        raw_output_layout=raw_output_layout,
        tensor_shape_list_fn=_tensor_shape_list,
        target_shape_literal_fn=_target_shape_literal,
    )
    _is_constant_tensor_name = lambda tensor_name: _is_constant_tensor_name_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _shape_tensor_constant_is_non_zero_int_vector = lambda tensor_name: _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _static_int_tensor_values = lambda tensor_name, *, _visited=None: _static_int_tensor_values_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        tensor_name=tensor_name,
        _visited=_visited,
    )
    _reshape_shape_tensor_uses_runtime_dims = lambda tensor_name, *, _visited=None: _reshape_shape_tensor_uses_runtime_dims_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        tensor_name=tensor_name,
        _visited=_visited,
    )
    _should_skip_align_for_shape_preserving_unary = lambda input_name, output_name: _should_skip_align_for_shape_preserving_unary_for_codegen(
        model_ir=model_ir,
        input_name=input_name,
        output_name=output_name,
        tensor_shape_list_fn=_tensor_shape_list,
    )
    _next_unique_attr_name = lambda base_name: _next_unique_attr_name_for_codegen(
        base_name=base_name,
        module_attr_counts=module_attr_counts,
        affine_layer_norm_specs=affine_layer_norm_specs,
        op_module_attr_names=op_module_attr_names,
    )
    _canonical_codegen_name = _canonical_codegen_name_for_codegen
    _match_affine_layer_norm = lambda op_index, op: _match_affine_layer_norm_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        is_constant_tensor_name_fn=_is_constant_tensor_name,
        canonical_codegen_name_fn=_canonical_codegen_name,
        next_unique_attr_name_fn=_next_unique_attr_name,
        op_index=op_index,
        op=op,
    )
    _match_swish_activation_pattern = lambda tensor_name, consumer_indices: _match_swish_activation_pattern_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        tensor_name=tensor_name,
        consumer_indices=consumer_indices,
    )
    _topk_codegen_layout_bridge = lambda *, input_name, value_output_name, index_output_name: _topk_codegen_layout_bridge_for_codegen(
        tensor_shape_list_fn=_tensor_shape_list,
        input_name=input_name,
        value_output_name=value_output_name,
        index_output_name=index_output_name,
    )
    _pad_literal_expr = lambda tensor_name: _pad_literal_expr_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _constant_pad_pairs = lambda tensor_name: _constant_pad_pairs_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _scalar_literal_expr = lambda tensor_name: _scalar_literal_expr_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _int_scalar_literal_expr = lambda tensor_name: _int_scalar_literal_expr_for_codegen(
        static_int_tensor_values_fn=lambda name: _static_int_tensor_values(name),
        tensor_name=tensor_name,
    )
    _axis_expr_from_input = lambda tensor_name, *, device_expr: _axis_expr_from_input_for_codegen(
        runtime_imports=runtime_imports,
        int_scalar_literal_expr_fn=_int_scalar_literal_expr,
        tensor_expr_fn=_tensor_expr,
        tensor_name=tensor_name,
        device_expr=device_expr,
    )
    _activation_lines = _activation_lines_for_codegen
    _static_mirror_pad_expr = lambda *, input_tensor_name, pads_tensor_name: _static_mirror_pad_expr_for_codegen(
        model_ir=model_ir,
        runtime_imports=runtime_imports,
        constant_pad_pairs_fn=_constant_pad_pairs,
        tensor_expr_fn=_tensor_expr,
        input_tensor_name=input_tensor_name,
        pads_tensor_name=pads_tensor_name,
    )
    _is_sequential_single_input_graph = lambda: _is_sequential_single_input_graph_for_codegen(
        model_ir=model_ir,
    )
    _is_channel_last_layout = _is_channel_last_layout_for_codegen
    _tensor_dtype_name = lambda tensor_name: _tensor_dtype_name_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _can_emit_direct_module_call = lambda op: _can_emit_direct_module_call_for_codegen(
        model_ir=model_ir,
        is_channel_last_layout_fn=_is_channel_last_layout,
        op=op,
    )
    _gather_input_pre_permute = lambda *, params_name, output_name, axis, batch_dims: _gather_input_pre_permute_for_codegen(
        model_ir=model_ir,
        params_name=params_name,
        output_name=output_name,
        axis=axis,
        batch_dims=batch_dims,
    )
    _conv2d_output_spatial_shape = lambda *, input_hw, kernel_hw, stride_hw, dilation_hw, padding_mode: _conv2d_output_spatial_shape_for_codegen(
        input_hw=input_hw,
        kernel_hw=kernel_hw,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        padding_mode=padding_mode,
    )
    _conv3d_output_spatial_shape = lambda *, input_dhw, kernel_dhw, stride_dhw, dilation_dhw, padding_mode: _conv3d_output_spatial_shape_for_codegen(
        input_dhw=input_dhw,
        kernel_dhw=kernel_dhw,
        stride_dhw=stride_dhw,
        dilation_dhw=dilation_dhw,
        padding_mode=padding_mode,
    )
    _conv3d_transpose_output_spatial_shape = lambda *, input_dhw, kernel_dhw, stride_dhw, dilation_dhw, padding_mode: _conv3d_transpose_output_spatial_shape_for_codegen(
        input_dhw=input_dhw,
        kernel_dhw=kernel_dhw,
        stride_dhw=stride_dhw,
        dilation_dhw=dilation_dhw,
        padding_mode=padding_mode,
    )
    _conv2d_same_pad_arg = lambda *, input_shape, output_shape, weight_shape, options, input_pre_permute=None, input_logical_layout=None, output_logical_layout=None: _conv2d_same_pad_padding_arg_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        options=options,
        input_pre_permute=input_pre_permute,
        input_logical_layout=input_logical_layout,
        output_logical_layout=output_logical_layout,
    )
    _infer_conv2d_layout_candidate = lambda *, input_shape, output_shape, weight_shape, options, depthwise: _infer_conv2d_layout_candidate_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        options=options,
        depthwise=depthwise,
    )
    _conv2d_input_pre_permute = lambda input_shape, output_shape, weight_shape, options, *, input_logical_layout=None, output_logical_layout=None, depthwise=False: _conv2d_input_pre_permute_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        options=options,
        input_logical_layout=input_logical_layout,
        output_logical_layout=output_logical_layout,
        depthwise=depthwise,
    )
    _infer_effective_rank4_runtime_layout = lambda tensor_name: _infer_effective_rank4_runtime_layout_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        consumer_index=consumer_index,
        tensor_name=tensor_name,
    )
    _infer_conv2d_ctor_params = lambda *, input_shape, output_shape, weight_shape, options, input_logical_layout=None, output_logical_layout=None, depthwise: _infer_conv2d_ctor_params_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        options=options,
        input_logical_layout=input_logical_layout,
        output_logical_layout=output_logical_layout,
        depthwise=depthwise,
    )
    _infer_conv3d_ctor_params = lambda *, input_shape, output_shape, weight_shape, options, input_logical_layout=None, output_logical_layout=None: _infer_conv3d_ctor_params_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        options=options,
        input_logical_layout=input_logical_layout,
        output_logical_layout=output_logical_layout,
    )
    _infer_conv3d_transpose_ctor_params = lambda *, input_shape, output_shape, weight_shape, options, input_logical_layout=None, output_logical_layout=None: _infer_conv3d_transpose_ctor_params_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        options=options,
        input_logical_layout=input_logical_layout,
        output_logical_layout=output_logical_layout,
    )

    for op_index, op in enumerate(model_ir.operators):
        affine_layer_norm_spec = _match_affine_layer_norm(int(op_index), op)
        if affine_layer_norm_spec is None:
            continue
        attr_name = str(affine_layer_norm_spec["attr_name"])
        module_init_lines.extend(
            [
                f"self.{attr_name} = _AffineLayerNorm(",
                f"    shape={repr(list(affine_layer_norm_spec['gamma_shape']))},",
                f"    dtype={_torch_dtype_literal(str(affine_layer_norm_spec['gamma_dtype']))},",
                ")",
            ]
        )
        load_specs.append((f"{attr_name}.gamma", str(affine_layer_norm_spec["gamma_name"])))
        load_specs.append((f"{attr_name}.beta", str(affine_layer_norm_spec["beta_name"])))
        submodule_state_tensor_names.update(
            {
                str(affine_layer_norm_spec["gamma_name"]),
                str(affine_layer_norm_spec["beta_name"]),
            }
        )
        affine_layer_norm_specs[int(op_index)] = affine_layer_norm_spec
        skipped_op_indices.add(int(affine_layer_norm_spec["mul_op_index"]))

    for op_index, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        if op_type not in _DIRECT_CODEGEN_MODULE_OP_TYPES:
            continue
        attr_base = _direct_codegen_module_attr_base(op_type)
        attr_index = int(module_attr_counts.get(attr_base, 0))
        module_attr_counts[attr_base] = attr_index + 1
        attr_name = f"{attr_base}_{attr_index}"
        op_module_attr_names[int(op_index)] = attr_name
        if op_type == "UNIDIRECTIONAL_SEQUENCE_RNN":
            if not _can_direct_codegen_sequence_rnn_op(model_ir, op):
                raise ModelIRPyTorchExportError(
                    "Native PyTorch-like model.py codegen does not support this UNIDIRECTIONAL_SEQUENCE_RNN configuration."
                )
            weight_name = _sequence_lstm_input_name(op, 1)
            recurrent_name = _sequence_lstm_input_name(op, 2)
            bias_name = _sequence_lstm_input_name(op, 3)
            weight = _require_constant_array_from_model_ir(
                model_ir=model_ir,
                tensor_name=weight_name,
                context="RNN input weight",
            )
            hidden_size = int(weight.shape[0])
            input_size = int(weight.shape[1])
            weight_dtype = str(model_ir.tensors[weight_name].dtype)
            bias_hh_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_bias_hh_l0",
                data=np.zeros((hidden_size,), dtype=np.float32),
                dtype=weight_dtype,
            )
            module_param_tensor_names.update(
                {
                    str(weight_name),
                    str(recurrent_name),
                    str(bias_name),
                    str(bias_hh_name),
                }
            )
            activation_name = str(op.options.get("fusedActivationFunction", "TANH")).strip().lower()
            module_init_lines.extend(
                [
                    f"self.{attr_name} = _SequenceRNNBlock(",
                    f"    input_size={input_size},",
                    f"    hidden_size={hidden_size},",
                    f"    activation={activation_name!r},",
                    ")",
                ]
            )
            load_specs.extend(
                [
                    (f"{attr_name}.rnn.weight_ih_l0", str(weight_name)),
                    (f"{attr_name}.rnn.weight_hh_l0", str(recurrent_name)),
                    (f"{attr_name}.rnn.bias_ih_l0", str(bias_name)),
                    (f"{attr_name}.rnn.bias_hh_l0", str(bias_hh_name)),
                ]
            )
            continue
        if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            if not _can_direct_codegen_sequence_lstm_op(model_ir, op):
                raise ModelIRPyTorchExportError(
                    "Native PyTorch-like model.py codegen does not support this UNIDIRECTIONAL_SEQUENCE_LSTM configuration."
                )
            index_spec = _sequence_lstm_index_spec(op)
            if index_spec is None:
                raise ModelIRPyTorchExportError(
                    "Native PyTorch-like model.py codegen could not resolve UNIDIRECTIONAL_SEQUENCE_LSTM input layout."
                )
            weight_input_names = [
                _sequence_lstm_input_name(op, idx)
                for idx in list(index_spec["weight_input_indices"])
            ]
            recurrent_input_names = [
                _sequence_lstm_input_name(op, idx)
                for idx in list(index_spec["recurrent_input_indices"])
            ]
            input_weight = _require_constant_array_from_model_ir(
                model_ir=model_ir,
                tensor_name=weight_input_names[0],
                context="LSTM input weight",
            )
            hidden_size = int(input_weight.shape[0])
            input_size = int(input_weight.shape[1])
            weight_dtype = str(model_ir.tensors[weight_input_names[0]].dtype)
            bias_ih_name = _sequence_lstm_bias_array_for_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                op=op,
                indices=list(index_spec["bias_indices"]),
                hidden_size=hidden_size,
                dtype=weight_dtype,
                base_name=f"{attr_name}_bias_ih_l0",
            )
            bias_hh_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_bias_hh_l0",
                data=np.zeros((4 * hidden_size,), dtype=np.float32),
                dtype=weight_dtype,
            )
            weight_ih_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_weight_ih_l0",
                data=np.concatenate(
                    [
                        _require_constant_array_from_model_ir(
                            model_ir=model_ir,
                            tensor_name=name,
                            context=f"LSTM input gate weight {gate_name}",
                        )
                        for gate_name, name in zip(["i", "f", "c", "o"], weight_input_names)
                    ],
                    axis=0,
                ).astype(np.float32, copy=False),
                dtype=weight_dtype,
            )
            weight_hh_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_weight_hh_l0",
                data=np.concatenate(
                    [
                        _require_constant_array_from_model_ir(
                            model_ir=model_ir,
                            tensor_name=name,
                            context=f"LSTM recurrent gate weight {gate_name}",
                        )
                        for gate_name, name in zip(["i", "f", "c", "o"], recurrent_input_names)
                    ],
                    axis=0,
                ).astype(np.float32, copy=False),
                dtype=weight_dtype,
            )
            module_param_tensor_names.update(
                {
                    str(weight_ih_name),
                    str(weight_hh_name),
                    str(bias_ih_name),
                    str(bias_hh_name),
                }
            )
            module_init_lines.extend(
                [
                    f"self.{attr_name} = _SequenceLSTMBlock(",
                    f"    input_size={input_size},",
                    f"    hidden_size={hidden_size},",
                    f"    sequence_length={_static_sequence_length(str(op.inputs[0]))},",
                    "    bidirectional=False,",
                    "    merge_outputs=True,",
                    ")",
                ]
            )
            load_specs.extend(
                [
                    (f"{attr_name}.lstm.weight_ih_l0", str(weight_ih_name)),
                    (f"{attr_name}.lstm.weight_hh_l0", str(weight_hh_name)),
                    (f"{attr_name}.lstm.bias_ih_l0", str(bias_ih_name)),
                    (f"{attr_name}.lstm.bias_hh_l0", str(bias_hh_name)),
                ]
            )
            continue
        if op_type == "BIDIRECTIONAL_SEQUENCE_LSTM":
            if not _can_direct_codegen_sequence_lstm_op(model_ir, op):
                raise ModelIRPyTorchExportError(
                    "Native PyTorch-like model.py codegen does not support this BIDIRECTIONAL_SEQUENCE_LSTM configuration."
                )
            index_spec = _sequence_lstm_index_spec(op)
            if index_spec is None:
                raise ModelIRPyTorchExportError(
                    "Native PyTorch-like model.py codegen could not resolve BIDIRECTIONAL_SEQUENCE_LSTM input layout."
                )
            fw_weight_input_names = [
                _sequence_lstm_input_name(op, idx)
                for idx in list(index_spec["fw_weight_input_indices"])
            ]
            fw_recurrent_input_names = [
                _sequence_lstm_input_name(op, idx)
                for idx in list(index_spec["fw_recurrent_input_indices"])
            ]
            bw_weight_input_names = [
                _sequence_lstm_input_name(op, idx)
                for idx in list(index_spec["bw_weight_input_indices"])
            ]
            bw_recurrent_input_names = [
                _sequence_lstm_input_name(op, idx)
                for idx in list(index_spec["bw_recurrent_input_indices"])
            ]
            fw_input_weight = _require_constant_array_from_model_ir(
                model_ir=model_ir,
                tensor_name=fw_weight_input_names[0],
                context="forward LSTM input weight",
            )
            hidden_size = int(fw_input_weight.shape[0])
            input_size = int(fw_input_weight.shape[1])
            weight_dtype = str(model_ir.tensors[fw_weight_input_names[0]].dtype)
            fw_bias_ih_name = _sequence_lstm_bias_array_for_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                op=op,
                indices=list(index_spec["fw_bias_indices"]),
                hidden_size=hidden_size,
                dtype=weight_dtype,
                base_name=f"{attr_name}_bias_ih_l0",
            )
            fw_bias_hh_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_bias_hh_l0",
                data=np.zeros((4 * hidden_size,), dtype=np.float32),
                dtype=weight_dtype,
            )
            bw_bias_ih_name = _sequence_lstm_bias_array_for_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                op=op,
                indices=list(index_spec["bw_bias_indices"]),
                hidden_size=hidden_size,
                dtype=weight_dtype,
                base_name=f"{attr_name}_bias_ih_l0_reverse",
            )
            bw_bias_hh_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_bias_hh_l0_reverse",
                data=np.zeros((4 * hidden_size,), dtype=np.float32),
                dtype=weight_dtype,
            )
            fw_weight_ih_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_weight_ih_l0",
                data=np.concatenate(
                    [
                        _require_constant_array_from_model_ir(
                            model_ir=model_ir,
                            tensor_name=name,
                            context=f"forward LSTM input gate weight {gate_name}",
                        )
                        for gate_name, name in zip(["i", "f", "c", "o"], fw_weight_input_names)
                    ],
                    axis=0,
                ).astype(np.float32, copy=False),
                dtype=weight_dtype,
            )
            fw_weight_hh_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_weight_hh_l0",
                data=np.concatenate(
                    [
                        _require_constant_array_from_model_ir(
                            model_ir=model_ir,
                            tensor_name=name,
                            context=f"forward LSTM recurrent gate weight {gate_name}",
                        )
                        for gate_name, name in zip(["i", "f", "c", "o"], fw_recurrent_input_names)
                    ],
                    axis=0,
                ).astype(np.float32, copy=False),
                dtype=weight_dtype,
            )
            bw_weight_ih_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_weight_ih_l0_reverse",
                data=np.concatenate(
                    [
                        _require_constant_array_from_model_ir(
                            model_ir=model_ir,
                            tensor_name=name,
                            context=f"backward LSTM input gate weight {gate_name}",
                        )
                        for gate_name, name in zip(["i", "f", "c", "o"], bw_weight_input_names)
                    ],
                    axis=0,
                ).astype(np.float32, copy=False),
                dtype=weight_dtype,
            )
            bw_weight_hh_name = _add_synthetic_tensor_to_model_ir(model_ir=model_ir, synthetic_tensor_serial_ref=synthetic_tensor_serial_ref, 
                base_name=f"{attr_name}_weight_hh_l0_reverse",
                data=np.concatenate(
                    [
                        _require_constant_array_from_model_ir(
                            model_ir=model_ir,
                            tensor_name=name,
                            context=f"backward LSTM recurrent gate weight {gate_name}",
                        )
                        for gate_name, name in zip(["i", "f", "c", "o"], bw_recurrent_input_names)
                    ],
                    axis=0,
                ).astype(np.float32, copy=False),
                dtype=weight_dtype,
            )
            module_param_tensor_names.update(
                {
                    str(fw_weight_ih_name),
                    str(fw_weight_hh_name),
                    str(fw_bias_ih_name),
                    str(fw_bias_hh_name),
                    str(bw_weight_ih_name),
                    str(bw_weight_hh_name),
                    str(bw_bias_ih_name),
                    str(bw_bias_hh_name),
                }
            )
            module_init_lines.extend(
                [
                    f"self.{attr_name} = _SequenceLSTMBlock(",
                    f"    input_size={input_size},",
                    f"    hidden_size={hidden_size},",
                    f"    sequence_length={_static_sequence_length(str(op.inputs[0]))},",
                    "    bidirectional=True,",
                    f"    merge_outputs={bool(op.options.get('mergeOutputs', True))},",
                    ")",
                ]
            )
            load_specs.extend(
                [
                    (f"{attr_name}.lstm.weight_ih_l0", str(fw_weight_ih_name)),
                    (f"{attr_name}.lstm.weight_hh_l0", str(fw_weight_hh_name)),
                    (f"{attr_name}.lstm.bias_ih_l0", str(fw_bias_ih_name)),
                    (f"{attr_name}.lstm.bias_hh_l0", str(fw_bias_hh_name)),
                    (f"{attr_name}.lstm.weight_ih_l0_reverse", str(bw_weight_ih_name)),
                    (f"{attr_name}.lstm.weight_hh_l0_reverse", str(bw_weight_hh_name)),
                    (f"{attr_name}.lstm.bias_ih_l0_reverse", str(bw_bias_ih_name)),
                    (f"{attr_name}.lstm.bias_hh_l0_reverse", str(bw_bias_hh_name)),
                ]
            )
            continue
        weight_name = str(op.inputs[1]) if len(op.inputs) >= 2 else ""
        bias_name = ""
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D"} and len(op.inputs) >= 3 and str(op.inputs[2]) != "":
            bias_name = str(op.inputs[2])
        elif op_type == "FULLY_CONNECTED" and len(op.inputs) >= 3 and str(op.inputs[2]) != "":
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
        conv_input_pre_permute: Optional[List[int]] = None
        conv_same_pad_arg: Optional[List[int]] = None
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            conv_input_pre_permute = _conv2d_input_pre_permute(
                _tensor_shape_list(str(input_tensor_name)),
                _tensor_shape_list(str(op.outputs[0])),
                _tensor_shape_list(str(op.inputs[1])),
                options,
                input_logical_layout=normalize_logical_layout(input_tensor.logical_layout),
                output_logical_layout=normalize_logical_layout(
                    model_ir.tensors[str(op.outputs[0])].logical_layout
                ),
                depthwise=(op_type == "DEPTHWISE_CONV_2D"),
            )
            conv_same_pad_arg = _conv2d_same_pad_arg(
                input_shape=_tensor_shape_list(str(input_tensor_name)),
                output_shape=_tensor_shape_list(str(op.outputs[0])),
                weight_shape=_tensor_shape_list(str(op.inputs[1])),
                options=options,
                input_pre_permute=conv_input_pre_permute,
                input_logical_layout=normalize_logical_layout(input_tensor.logical_layout),
                output_logical_layout=normalize_logical_layout(
                    model_ir.tensors[str(op.outputs[0])].logical_layout
                ),
            )
            conv_module_pad_specs[int(op_index)] = conv_same_pad_arg
        fused_block_input_name = str(input_tensor_name)
        fused_block_output_name = str(op.outputs[0]) if len(op.outputs) == 1 else ""
        fused_block_pad: Optional[str] = None
        fused_block_pad_mode = "constant"
        fused_block_pad_value: Optional[str] = None
        fused_bridge_output_op_idx: Optional[int] = None
        fused_block_activation, fused_block_negative_slope = _conv_block_activation_config_from_fused_name(
            str(op.options.get("fusedActivationFunction", "NONE")),
            alpha=(
                float(op.options.get("alpha", 0.2))
                if "alpha" in op.options
                else None
            ),
        )
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"} and len(op.outputs) == 1:
            (
                fused_block_input_name,
                conv_input_pre_permute,
                folded_public_bridge_op_idx,
            ) = _fold_single_consumer_public_input_bridge(
                tensor_name=str(fused_block_input_name),
                downstream_permute=conv_input_pre_permute,
            )
            if folded_public_bridge_op_idx is not None:
                skipped_op_indices.add(int(folded_public_bridge_op_idx))
            input_producer_idx = producer_index.get(str(fused_block_input_name), None)
            if input_producer_idx is not None:
                input_producer = model_ir.operators[int(input_producer_idx)]
                input_producer_type = str(input_producer.op_type)
                if (
                    input_producer_type in {"PAD", "PADV2"}
                    and len(consumer_index.get(str(fused_block_input_name), [])) == 1
                    and len(input_producer.inputs) >= 2
                ):
                    pad_axis_permutation = (
                        [int(v) for v in list(conv_input_pre_permute)]
                        if conv_input_pre_permute is not None
                        else None
                    )
                    if pad_axis_permutation is None:
                        pad_input_tensor = model_ir.tensors.get(str(input_producer.inputs[0]), None)
                        pad_output_tensor = model_ir.tensors.get(str(input_producer.outputs[0]), None)
                        if pad_input_tensor is not None and pad_output_tensor is not None:
                            pad_input_layout = normalize_logical_layout(pad_input_tensor.logical_layout)
                            pad_output_layout = normalize_logical_layout(pad_output_tensor.logical_layout)
                            pad_rank = len(list(pad_output_tensor.shape))
                            inferred_pad_perm = logical_layout_permutation(
                                source_layout=pad_input_layout,
                                target_layout=pad_output_layout,
                            )
                            if inferred_pad_perm is not None and len(inferred_pad_perm) == pad_rank:
                                pad_axis_permutation = [int(v) for v in list(inferred_pad_perm)]
                    original_conv_input_pre_permute = (
                        [int(v) for v in list(conv_input_pre_permute)]
                        if conv_input_pre_permute is not None
                        else None
                    )
                    pad_literal = _torch_pad_literal_for_constant_tensor(
                        model_ir.tensors.get(str(input_producer.inputs[1]), None),
                        axis_permutation=pad_axis_permutation,
                    )
                    if pad_literal is not None:
                        fused_block_input_name = str(input_producer.inputs[0])
                        fused_block_pad = pad_literal
                        pad_matches_pre_permuted_input = _pad_output_matches_pre_permuted_input(
                            input_tensor=model_ir.tensors.get(str(fused_block_input_name), None),
                            output_tensor=model_ir.tensors.get(str(input_producer.outputs[0]), None),
                            pads_tensor=model_ir.tensors.get(str(input_producer.inputs[1]), None),
                            input_pre_permute=conv_input_pre_permute,
                        )
                        if pad_matches_pre_permuted_input:
                            rewrite_pad_axis_permutation = original_conv_input_pre_permute
                            if rewrite_pad_axis_permutation is None:
                                pad_input_tensor = model_ir.tensors.get(str(input_producer.inputs[0]), None)
                                pad_output_tensor = model_ir.tensors.get(str(input_producer.outputs[0]), None)
                                if pad_input_tensor is not None and pad_output_tensor is not None:
                                    pad_input_layout = normalize_logical_layout(pad_input_tensor.logical_layout)
                                    pad_output_layout = normalize_logical_layout(pad_output_tensor.logical_layout)
                                    pad_rank = len(list(pad_output_tensor.shape))
                                    inferred_pad_perm = logical_layout_permutation(
                                        source_layout=pad_output_layout,
                                        target_layout=pad_input_layout,
                                    )
                                    if inferred_pad_perm is not None and len(inferred_pad_perm) == pad_rank:
                                        rewrite_pad_axis_permutation = [int(v) for v in list(inferred_pad_perm)]
                            if rewrite_pad_axis_permutation is not None:
                                rewritten_pad_literal = _torch_pad_literal_for_constant_tensor(
                                    model_ir.tensors.get(str(input_producer.inputs[1]), None),
                                    axis_permutation=rewrite_pad_axis_permutation,
                                )
                                if rewritten_pad_literal is not None:
                                    pad_literal = rewritten_pad_literal
                                    fused_block_pad = pad_literal
                            conv_input_pre_permute = None
                        upstream_producer_idx = producer_index.get(str(fused_block_input_name), None)
                        if upstream_producer_idx is not None:
                            upstream_producer = model_ir.operators[int(upstream_producer_idx)]
                            if (
                                str(upstream_producer.op_type) == "TRANSPOSE"
                                and len(upstream_producer.outputs) == 1
                                and len(consumer_index.get(str(fused_block_input_name), [])) == 1
                            ):
                                upstream_perm = _read_transpose_perm(model_ir, upstream_producer)
                                composed_perm = (
                                    None
                                    if pad_matches_pre_permuted_input
                                    else _compose_axis_permutations(
                                        upstream_perm,
                                        conv_input_pre_permute,
                                    )
                                )
                                if upstream_perm is not None or conv_input_pre_permute is not None:
                                    fused_block_input_name = str(upstream_producer.inputs[0])
                                    conv_input_pre_permute = composed_perm
                                    skipped_op_indices.add(int(upstream_producer_idx))
                        if input_producer_type == "PADV2" and len(input_producer.inputs) >= 3:
                            scalar_literal = _scalar_literal_for_constant_tensor(
                                model_ir.tensors.get(str(input_producer.inputs[2]), None)
                            )
                            if scalar_literal is not None:
                                fused_block_pad_value = scalar_literal
                        skipped_op_indices.add(int(input_producer_idx))
                        inlined_constant_tensor_names.add(str(input_producer.inputs[1]))
                        if input_producer_type == "PADV2" and len(input_producer.inputs) >= 3:
                            inlined_constant_tensor_names.add(str(input_producer.inputs[2]))
            bridged_output = _match_single_consumer_layout_bridge_transpose(
                tensor_name=str(op.outputs[0]),
                required_output_layout=channel_first_logical_layout(
                    len(list(model_ir.tensors[str(op.outputs[0])].shape))
                ),
            )
            effective_output_name = str(op.outputs[0])
            output_consumer_indices = consumer_index.get(str(op.outputs[0]), [])
            if bridged_output is not None:
                candidate_output_name, candidate_bridge_output_op_idx = bridged_output
                candidate_output_consumer_indices = consumer_index.get(str(candidate_output_name), [])
                can_fold_bridge_output = False
                candidate_swish_match = _match_swish_activation_pattern(
                    str(candidate_output_name),
                    candidate_output_consumer_indices,
                )
                if len(candidate_output_consumer_indices) == 1:
                    candidate_activation_op = model_ir.operators[int(candidate_output_consumer_indices[0])]
                    candidate_activation_type = str(candidate_activation_op.op_type)
                    can_fold_bridge_output = (
                        len(candidate_activation_op.inputs) == 1
                        and len(candidate_activation_op.outputs) == 1
                        and candidate_activation_type in {
                            "LEAKY_RELU",
                            "LOGISTIC",
                            "RELU",
                            "RELU6",
                            "RELU_N1_TO_1",
                            "RELU_0_TO_1",
                            "TANH",
                        }
                    )
                elif candidate_swish_match is not None and fused_block_activation == "none":
                    can_fold_bridge_output = True
                elif str(candidate_output_name) in {str(name) for name in list(model_ir.outputs)}:
                    can_fold_bridge_output = True
                if can_fold_bridge_output:
                    effective_output_name = str(candidate_output_name)
                    fused_bridge_output_op_idx = int(candidate_bridge_output_op_idx)
                    output_consumer_indices = candidate_output_consumer_indices
            swish_match = (
                _match_swish_activation_pattern(
                    str(effective_output_name),
                    output_consumer_indices,
                )
                if fused_block_activation == "none"
                else None
            )
            if swish_match is not None:
                fused_block_activation = "silu"
                fused_block_output_name = str(swish_match[0])
                for matched_op_idx in sorted(int(idx) for idx in swish_match[1]):
                    skipped_op_indices.add(int(matched_op_idx))
                if fused_bridge_output_op_idx is not None:
                    skipped_op_indices.add(int(fused_bridge_output_op_idx))
            elif len(output_consumer_indices) == 1:
                activation_op = model_ir.operators[int(output_consumer_indices[0])]
                activation_type = str(activation_op.op_type)
                if activation_type in {
                    "LEAKY_RELU",
                    "LOGISTIC",
                    "RELU",
                    "RELU6",
                    "RELU_N1_TO_1",
                    "RELU_0_TO_1",
                    "TANH",
                } and len(activation_op.inputs) == 1 and len(activation_op.outputs) == 1 and fused_block_activation == "none":
                    fused_block_activation, fused_block_negative_slope = _conv_block_activation_config(activation_op)
                    fused_block_output_name = str(activation_op.outputs[0])
                    skipped_op_indices.add(int(output_consumer_indices[0]))
                    if fused_bridge_output_op_idx is not None:
                        skipped_op_indices.add(int(fused_bridge_output_op_idx))
            elif fused_bridge_output_op_idx is not None:
                fused_block_output_name = str(effective_output_name)
                skipped_op_indices.add(int(fused_bridge_output_op_idx))
        use_conv_block = op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
        if use_conv_block:
            attr_base = "conv_block"
            attr_index = int(module_attr_counts.get(attr_base, 0))
            module_attr_counts[attr_base] = attr_index + 1
            attr_name = f"{attr_base}_{attr_index}"
            op_module_attr_names[int(op_index)] = attr_name
            fused_module_specs[int(op_index)] = {
                "input_name": str(fused_block_input_name),
                "output_name": str(fused_block_output_name),
                "input_pre_permute": conv_input_pre_permute,
                "pad": fused_block_pad if fused_block_pad is not None else (repr(conv_same_pad_arg) if conv_same_pad_arg is not None else None),
                "pad_mode": fused_block_pad_mode,
                "pad_value": fused_block_pad_value,
                "activation": fused_block_activation,
                "negative_slope": fused_block_negative_slope,
            }
        if op_type == "CONV_2D":
            conv_padding = (
                [0, 0]
                if conv_same_pad_arg is not None
                else [
                    int(((int(weight_tensor.shape[2]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                    int(((int(weight_tensor.shape[3]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                ]
            )
            conv_in_channels, conv_groups = _infer_conv2d_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
                input_logical_layout=input_tensor.logical_layout,
                output_logical_layout=model_ir.tensors[str(op.outputs[0])].logical_layout,
                depthwise=False,
            )
            conv_ctor_lines = [
                "torch.nn.Conv2d(",
                f"    in_channels={conv_in_channels},",
                f"    out_channels={int(weight_tensor.shape[0])},",
                f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                f"    padding={_shape_literal(conv_padding)},",
                f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                f"    groups={conv_groups},",
                f"    bias={str(bias_name != '')},",
                ")",
            ]
            if use_conv_block:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = _Conv2dBlock(",
                        f"    in_channels={conv_in_channels},",
                        f"    out_channels={int(weight_tensor.shape[0])},",
                        f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                        f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                        f"    padding={_shape_literal(conv_padding)},",
                        f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                        f"    groups={conv_groups},",
                        f"    bias={str(bias_name != '')},",
                        f"    pad={fused_block_pad if fused_block_pad is not None else (repr(conv_same_pad_arg) if conv_same_pad_arg is not None else 'None')},",
                        f"    activation={fused_block_activation!r},",
                        f"    negative_slope={repr(fused_block_negative_slope if fused_block_negative_slope is not None else 0.2)},",
                        f"    pad_mode={fused_block_pad_mode!r},",
                        f"    pad_value={fused_block_pad_value if fused_block_pad_value is not None else '0.0'},",
                        ")",
                    ]
                )
            else:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = {conv_ctor_lines[0]}",
                        *conv_ctor_lines[1:],
                    ]
                )
        elif op_type == "DEPTHWISE_CONV_2D":
            conv_padding = (
                [0, 0]
                if conv_same_pad_arg is not None
                else [
                    int(((int(weight_tensor.shape[2]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                    int(((int(weight_tensor.shape[3]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                ]
            )
            depthwise_in_channels, depthwise_groups = _infer_conv2d_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
                input_logical_layout=input_tensor.logical_layout,
                output_logical_layout=model_ir.tensors[str(op.outputs[0])].logical_layout,
                depthwise=True,
            )
            conv_ctor_lines = [
                "torch.nn.Conv2d(",
                f"    in_channels={depthwise_in_channels},",
                f"    out_channels={int(weight_tensor.shape[0])},",
                f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                f"    padding={_shape_literal(conv_padding)},",
                f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                f"    groups={depthwise_groups},",
                f"    bias={str(bias_name != '')},",
                ")",
            ]
            if use_conv_block:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = _Conv2dBlock(",
                        f"    in_channels={depthwise_in_channels},",
                        f"    out_channels={int(weight_tensor.shape[0])},",
                        f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                        f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                        f"    padding={_shape_literal(conv_padding)},",
                        f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                        f"    groups={depthwise_groups},",
                        f"    bias={str(bias_name != '')},",
                        f"    pad={fused_block_pad if fused_block_pad is not None else (repr(conv_same_pad_arg) if conv_same_pad_arg is not None else 'None')},",
                        f"    activation={fused_block_activation!r},",
                        f"    negative_slope={repr(fused_block_negative_slope if fused_block_negative_slope is not None else 0.2)},",
                        f"    pad_mode={fused_block_pad_mode!r},",
                        f"    pad_value={fused_block_pad_value if fused_block_pad_value is not None else '0.0'},",
                        ")",
                    ]
                )
            else:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = {conv_ctor_lines[0]}",
                        *conv_ctor_lines[1:],
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
            conv3d_in_channels, conv3d_groups, conv3d_out_channels, conv3d_kernel_size = _infer_conv3d_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
                input_logical_layout=normalize_logical_layout(input_tensor.logical_layout),
                output_logical_layout=normalize_logical_layout(
                    model_ir.tensors[str(op.outputs[0])].logical_layout
                ),
            )
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.Conv3d(",
                    f"    in_channels={conv3d_in_channels},",
                    f"    out_channels={conv3d_out_channels},",
                    f"    kernel_size={_shape_literal(conv3d_kernel_size)},",
                    f"    stride={_shape_literal([int(options.get('strideD', 1)), int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(conv3d_kernel_size[0]) - 1) * int(options.get('dilationDFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3d_kernel_size[1]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3d_kernel_size[2]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    dilation={_shape_literal([int(options.get('dilationDFactor', 1)), int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                    f"    groups={conv3d_groups},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "CONV_3D_TRANSPOSE":
            conv3dt_in_channels, conv3dt_out_channels, conv3dt_kernel_size, conv3dt_groups = _infer_conv3d_transpose_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
                input_logical_layout=normalize_logical_layout(input_tensor.logical_layout),
                output_logical_layout=normalize_logical_layout(
                    model_ir.tensors[str(op.outputs[0])].logical_layout
                ),
            )
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.ConvTranspose3d(",
                    f"    in_channels={conv3dt_in_channels},",
                    f"    out_channels={conv3dt_out_channels},",
                    f"    kernel_size={_shape_literal(conv3dt_kernel_size)},",
                    f"    stride={_shape_literal([int(options.get('strideD', 1)), int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(conv3dt_kernel_size[0]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3dt_kernel_size[1]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3dt_kernel_size[2]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    groups={conv3dt_groups},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "FULLY_CONNECTED":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.Linear(",
                    f"    in_features={int(weight_tensor.shape[1])},",
                    f"    out_features={int(weight_tensor.shape[0])},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "PRELU":
            num_parameters = 1
            if isinstance(weight_tensor.data, np.ndarray):
                num_parameters = max(1, int(np.asarray(weight_tensor.data).size))
            elif len(list(weight_tensor.shape)) > 0:
                shape_values = [int(v) for v in list(weight_tensor.shape) if int(v) > 0]
                if len(shape_values) > 0:
                    num_parameters = max(1, int(np.prod(shape_values, dtype=np.int64)))
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.PReLU(",
                    f"    num_parameters={num_parameters},",
                    "    init=0.0,",
                    ")",
                ]
            )
        load_specs.append((f"{attr_name}.conv.weight" if use_conv_block else f"{attr_name}.weight", str(weight_name)))
        if bias_name != "":
            load_specs.append((f"{attr_name}.conv.bias" if use_conv_block else f"{attr_name}.bias", str(bias_name)))

    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type not in {"PAD", "PADV2", "MIRROR_PAD"}:
            continue
        if len(op.inputs) >= 2:
            paddings_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if paddings_tensor is not None and isinstance(paddings_tensor.data, np.ndarray):
                inlined_constant_tensor_names.add(str(op.inputs[1]))
        if op_type == "PADV2" and len(op.inputs) >= 3:
            value_tensor = model_ir.tensors.get(str(op.inputs[2]), None)
            if value_tensor is not None and isinstance(value_tensor.data, np.ndarray):
                inlined_constant_tensor_names.add(str(op.inputs[2]))
    for tensor_name, tensor in model_ir.tensors.items():
        if str(tensor_name) in module_param_tensor_names:
            continue
        if _is_small_inline_constant_tensor(tensor):
            inlined_constant_tensor_names.add(str(tensor_name))

    buffer_attr_names = _build_buffer_attr_name_map(
        model_ir=model_ir,
        tensor_storage_name_map=tensor_storage_name_map,
        excluded_tensor_names=module_param_tensor_names | submodule_state_tensor_names | inlined_constant_tensor_names,
    )
    buffer_attr_name_to_tensor_name = {
        str(attr_name): str(tensor_name) for tensor_name, attr_name in buffer_attr_names.items()
    }
    buffer_init_lines: List[str] = []
    for tensor_name, attr_name in buffer_attr_names.items():
        tensor = model_ir.tensors[str(tensor_name)]
        dtype_name = str(tensor.dtype).upper()
        shape_values = [int(v) for v in list(tensor.shape)]
        if bool(tensor.is_variable):
            buffer_init_lines.append(
                f"self.register_parameter({attr_name!r}, torch.nn.Parameter(torch.zeros({repr(shape_values)}, dtype={_torch_dtype_literal(dtype_name)}), requires_grad=False))"
            )
        else:
            buffer_init_lines.append(
                f"self.register_buffer({attr_name!r}, torch.zeros({repr(shape_values)}, dtype={_torch_dtype_literal(dtype_name)}), persistent=True)"
            )
        load_specs.append((str(attr_name), str(tensor_name)))

    _binary_trailing_axis_constant_buffer_alias_shape = lambda tensor_name, other_tensor_name: _binary_trailing_axis_constant_buffer_alias_shape_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        inlined_constant_tensor_names=inlined_constant_tensor_names,
        tensor_name=tensor_name,
        other_tensor_name=other_tensor_name,
    )
    _channel_first_rank4_constant_buffer_alias_shape = lambda tensor_name: _channel_first_rank4_constant_buffer_alias_shape_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        inlined_constant_tensor_names=inlined_constant_tensor_names,
        tensor_name=tensor_name,
    )
    _constant_permute_for_broadcast = lambda tensor_name, other_tensor_name: _constant_permute_for_broadcast_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
        other_tensor_name=other_tensor_name,
    )

    binary_constant_buffer_alias_attr_names: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    binary_constant_buffer_alias_exprs: Dict[Tuple[str, str], str] = {}
    channel_first_constant_buffer_alias_attr_names: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    channel_first_constant_buffer_alias_exprs: Dict[str, str] = {}
    channel_first_constant_buffer_alias_refresh_specs: List[Tuple[str, str, List[int]]] = []
    permuted_constant_buffer_alias_attr_names: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    permuted_constant_buffer_alias_exprs: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    permuted_constant_buffer_alias_refresh_specs: List[Tuple[str, str, List[int]]] = []
    transposed_constant_buffer_alias_attr_names: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    transposed_constant_buffer_alias_exprs: Dict[str, str] = {}
    transposed_constant_buffer_alias_refresh_specs: List[Tuple[str, str]] = []
    used_buffer_attr_names: Set[str] = set(buffer_attr_names.values())
    for op in model_ir.operators:
        if str(op.op_type) not in _DIRECT_CODEGEN_BINARY_FUNCTIONS:
            continue
        input_names = [str(name) for name in list(op.inputs)]
        if len(input_names) < 2:
            continue
        lhs_name = str(input_names[0])
        rhs_name = str(input_names[1])
        for constant_name, other_name in ((lhs_name, rhs_name), (rhs_name, lhs_name)):
            tensor = model_ir.tensors.get(str(constant_name), None)
            if tensor is None:
                continue
            dtype_name = str(tensor.dtype).upper()
            alias_shape = _binary_trailing_axis_constant_buffer_alias_shape(
                constant_name,
                other_name,
            )
            if alias_shape is not None:
                alias_key = (str(constant_name), tuple(int(v) for v in alias_shape))
                attr_name = binary_constant_buffer_alias_attr_names.get(alias_key, None)
                if attr_name is None:
                    storage_name = tensor_storage_name_map.get(str(constant_name), str(constant_name))
                    shape_suffix = "x".join(str(int(v)) for v in alias_shape)
                    base_attr_name = _sanitize_python_identifier(
                        f"const_{storage_name}_broadcast_{shape_suffix}",
                        prefix="const",
                    )
                    attr_name = _make_unique_identifier(base_attr_name, used_buffer_attr_names)
                    binary_constant_buffer_alias_attr_names[alias_key] = str(attr_name)
                    if bool(tensor.is_variable):
                        buffer_init_lines.append(
                            f"self.register_parameter({attr_name!r}, torch.nn.Parameter(torch.zeros({repr(alias_shape)}, dtype={_torch_dtype_literal(dtype_name)}), requires_grad=False))"
                        )
                    else:
                        buffer_init_lines.append(
                            f"self.register_buffer({attr_name!r}, torch.zeros({repr(alias_shape)}, dtype={_torch_dtype_literal(dtype_name)}), persistent=True)"
                        )
                    load_specs.append((str(attr_name), str(constant_name)))
                binary_constant_buffer_alias_exprs[(str(constant_name), str(other_name))] = f"self.{attr_name}"
            channel_first_alias_shape = _channel_first_rank4_constant_buffer_alias_shape(
                constant_name,
            )
            if channel_first_alias_shape is None:
                source_attr_name = buffer_attr_names.get(str(constant_name), None)
            else:
                source_attr_name = buffer_attr_names.get(str(constant_name), None)
                if source_attr_name is not None:
                    alias_key = (str(constant_name), tuple(int(v) for v in channel_first_alias_shape))
                    channel_first_attr_name = channel_first_constant_buffer_alias_attr_names.get(alias_key, None)
                    if channel_first_attr_name is None:
                        storage_name = tensor_storage_name_map.get(str(constant_name), str(constant_name))
                        shape_suffix = "x".join(str(int(v)) for v in channel_first_alias_shape)
                        base_attr_name = _shorten_generated_python_identifier(
                            f"const_{storage_name}_channel_first_{shape_suffix}",
                            prefix="const",
                        )
                        channel_first_attr_name = _make_unique_identifier(base_attr_name, used_buffer_attr_names)
                        channel_first_constant_buffer_alias_attr_names[alias_key] = str(channel_first_attr_name)
                        buffer_init_lines.append(
                            f"self.register_buffer({channel_first_attr_name!r}, torch.zeros({repr(channel_first_alias_shape)}, dtype={_torch_dtype_literal(dtype_name)}), persistent=False)"
                        )
                        channel_first_constant_buffer_alias_refresh_specs.append(
                            (
                                str(channel_first_attr_name),
                                str(source_attr_name),
                                [int(v) for v in channel_first_alias_shape],
                            )
                        )
                    channel_first_constant_buffer_alias_exprs[str(constant_name)] = f"self.{channel_first_attr_name}"

            source_attr_name = buffer_attr_names.get(str(constant_name), None)
            permute_for_broadcast = _constant_permute_for_broadcast(constant_name, other_name)
            if source_attr_name is not None and permute_for_broadcast is not None:
                tensor_shape = [int(v) for v in list(tensor.shape)]
                permuted_shape = [int(tensor_shape[int(idx)]) for idx in list(permute_for_broadcast)]
                alias_key = (str(constant_name), tuple(int(v) for v in list(permute_for_broadcast)))
                permuted_attr_name = permuted_constant_buffer_alias_attr_names.get(alias_key, None)
                if permuted_attr_name is None:
                    storage_name = tensor_storage_name_map.get(str(constant_name), str(constant_name))
                    perm_suffix = "_".join(str(int(v)) for v in list(permute_for_broadcast))
                    shape_suffix = "x".join(str(int(v)) for v in list(permuted_shape))
                    base_attr_name = _shorten_generated_python_identifier(
                        f"const_{storage_name}_perm_{perm_suffix}_{shape_suffix}",
                        prefix="const",
                    )
                    permuted_attr_name = _make_unique_identifier(base_attr_name, used_buffer_attr_names)
                    permuted_constant_buffer_alias_attr_names[alias_key] = str(permuted_attr_name)
                    buffer_init_lines.append(
                        f"self.register_buffer({permuted_attr_name!r}, torch.zeros({repr(permuted_shape)}, dtype={_torch_dtype_literal(dtype_name)}), persistent=False)"
                    )
                    permuted_constant_buffer_alias_refresh_specs.append(
                        (
                            str(permuted_attr_name),
                            str(source_attr_name),
                            [int(v) for v in list(permute_for_broadcast)],
                        )
                    )
                permuted_constant_buffer_alias_exprs[(str(constant_name), tuple(int(v) for v in list(permute_for_broadcast)))] = f"self.{permuted_attr_name}"
    for op in model_ir.operators:
        if str(op.op_type) != "BATCH_MATMUL" or len(op.inputs) < 2:
            continue
        input_names = [str(name) for name in list(op.inputs[:2])]
        transpose_flags = [
            bool(op.options.get("adjX", False)),
            bool(op.options.get("adjY", False)),
        ]
        for input_name, requires_transpose in zip(input_names, transpose_flags):
            if not bool(requires_transpose):
                continue
            tensor = model_ir.tensors.get(str(input_name), None)
            if (
                tensor is None
                or not isinstance(tensor.data, np.ndarray)
                or bool(tensor.is_variable)
                or str(input_name) in model_ir.inputs
                or str(input_name) in producer_index
                or str(input_name) in inlined_constant_tensor_names
            ):
                continue
            tensor_shape = [int(v) for v in list(tensor.shape)]
            if len(tensor_shape) < 2:
                continue
            source_attr_name = buffer_attr_names.get(str(input_name), None)
            if source_attr_name is None:
                continue
            alias_shape = list(tensor_shape[:-2]) + [int(tensor_shape[-1]), int(tensor_shape[-2])]
            alias_key = (str(input_name), tuple(int(v) for v in alias_shape))
            attr_name = transposed_constant_buffer_alias_attr_names.get(alias_key, None)
            if attr_name is None:
                dtype_name = str(tensor.dtype).upper()
                storage_name = tensor_storage_name_map.get(str(input_name), str(input_name))
                shape_suffix = "x".join(str(int(v)) for v in alias_shape)
                base_attr_name = _shorten_generated_python_identifier(
                    f"const_{storage_name}_transpose_last_two_{shape_suffix}",
                    prefix="const",
                )
                attr_name = _make_unique_identifier(base_attr_name, used_buffer_attr_names)
                transposed_constant_buffer_alias_attr_names[alias_key] = str(attr_name)
                buffer_init_lines.append(
                    f"self.register_buffer({attr_name!r}, torch.zeros({repr(alias_shape)}, dtype={_torch_dtype_literal(dtype_name)}), persistent=False)"
                )
                transposed_constant_buffer_alias_refresh_specs.append(
                    (str(attr_name), str(source_attr_name))
                )
            transposed_constant_buffer_alias_exprs[str(input_name)] = f"self.{attr_name}"

    _tensor_expr = lambda tensor_name: _tensor_expr_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        tensor_expr_aliases=tensor_expr_aliases,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        buffer_attr_names=buffer_attr_names,
        runtime_imports=runtime_imports,
        tensor_var_names=tensor_var_names,
        tensor_name=tensor_name,
    )
    _derived_local_var_name = lambda base_name, prefix="t": _derived_local_var_name_for_codegen(
        synthetic_local_var_names=synthetic_local_var_names,
        used_local_var_names=used_local_var_names,
        base_name=base_name,
        prefix=prefix,
    )

    _tensor_expr_for_channel_first_bridge = lambda tensor_name, perm: _tensor_expr_for_channel_first_bridge_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        tensor_name=tensor_name,
        perm=perm,
    )
    _channel_first_reduction_plan = lambda op, input_name: _channel_first_reduction_plan_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        op=op,
        input_name=input_name,
    )
    _normalized_constant_reduction_axes = lambda axis_values, rank: _normalized_constant_reduction_axes_for_codegen(
        axis_values=axis_values,
        rank=rank,
    )
    _direct_mean_reduction_expr = lambda *, input_expr, axes, input_rank, keepdims: _direct_mean_reduction_expr_for_codegen(
        normalized_constant_reduction_axes_fn=_normalized_constant_reduction_axes,
        input_expr=input_expr,
        axes=axes,
        input_rank=input_rank,
        keepdims=keepdims,
    )

    _channel_first_passthrough_input_expr = lambda tensor_name: _channel_first_passthrough_input_expr_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        tensor_expr_fn=_tensor_expr,
        tensor_name=tensor_name,
    )
    _can_resolve_channel_first_expr_statically = lambda tensor_name, seen_names=None: _can_resolve_channel_first_expr_statically_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        direct_codegen_unary_expressions=_DIRECT_CODEGEN_UNARY_EXPRESSIONS,
        tensor_name=tensor_name,
        seen_names=seen_names,
    )
    _can_emit_channel_first_shape_preserving_unary_op = lambda op: _can_emit_channel_first_shape_preserving_unary_op_for_codegen(
        model_ir=model_ir,
        direct_codegen_unary_expressions=_DIRECT_CODEGEN_UNARY_EXPRESSIONS,
        tensor_shape_list_fn=_tensor_shape_list,
        can_resolve_channel_first_expr_statically_fn=lambda tensor_name: _can_resolve_channel_first_expr_statically(tensor_name),
        op=op,
    )

    _all_consumers_are_channel_first_binary_ops = lambda output_name: _all_consumers_are_channel_first_binary_ops_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        direct_codegen_binary_functions=_DIRECT_CODEGEN_BINARY_FUNCTIONS,
        can_emit_channel_first_binary_op_fn=lambda op: _can_emit_channel_first_binary_op(op),
        output_name=output_name,
    )
    _can_omit_materialized_channel_last_alias_recursive = lambda output_name, seen_names: _can_omit_materialized_channel_last_alias_recursive_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        direct_codegen_unary_expressions=_DIRECT_CODEGEN_UNARY_EXPRESSIONS,
        tensor_shape_list_fn=_tensor_shape_list,
        channel_first_reduction_plan_fn=_channel_first_reduction_plan,
        can_emit_channel_first_shape_preserving_unary_op_fn=_can_emit_channel_first_shape_preserving_unary_op,
        can_emit_channel_first_binary_op_fn=lambda op: _can_emit_channel_first_binary_op(op),
        can_resolve_channel_first_expr_statically_fn=lambda tensor_name: _can_resolve_channel_first_expr_statically(tensor_name),
        conv2d_input_pre_permute_fn=_conv2d_input_pre_permute,
        output_name=output_name,
        seen_names=seen_names,
    )
    _can_omit_materialized_channel_last_alias = lambda output_name: _can_omit_materialized_channel_last_alias_recursive(str(output_name), set())

    _channel_first_binary_input_expr = lambda tensor_name, other_tensor_name: _channel_first_binary_input_expr_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        channel_first_constant_buffer_alias_exprs=channel_first_constant_buffer_alias_exprs,
        permuted_constant_buffer_alias_exprs=permuted_constant_buffer_alias_exprs,
        scalar_literal_expr_fn=_scalar_literal_expr,
        tensor_shape_list_fn=_tensor_shape_list,
        tensor_expr_fn=_tensor_expr,
        channel_first_passthrough_input_expr_fn=_channel_first_passthrough_input_expr,
        tensor_name=tensor_name,
        other_tensor_name=other_tensor_name,
    )

    _channel_first_constant_expr_for_buffer_attr = lambda buffer_expr, target_shape: _channel_first_constant_expr_for_buffer_attr_for_codegen(
        buffer_attr_name_to_tensor_name=buffer_attr_name_to_tensor_name,
        channel_first_constant_buffer_alias_exprs=channel_first_constant_buffer_alias_exprs,
        channel_first_rank4_constant_buffer_alias_shape_fn=_channel_first_rank4_constant_buffer_alias_shape,
        buffer_expr=buffer_expr,
        target_shape=target_shape,
    )
    _permuted_constant_expr_for_tensor_name = lambda tensor_name, perm: _permuted_constant_expr_for_tensor_name_for_codegen(
        permuted_constant_buffer_alias_exprs=permuted_constant_buffer_alias_exprs,
        tensor_name=tensor_name,
        perm=perm,
    )
    _transposed_constant_expr_for_tensor_name = lambda tensor_name: _transposed_constant_expr_for_tensor_name_for_codegen(
        transposed_constant_buffer_alias_exprs=transposed_constant_buffer_alias_exprs,
        tensor_name=tensor_name,
    )

    _can_emit_channel_first_binary_op = lambda op: _can_emit_channel_first_binary_op_for_codegen(
        model_ir=model_ir,
        tensor_shape_list_fn=_tensor_shape_list,
        channel_first_shape_for_tensor_fn=_channel_first_shape_for_tensor,
        scalar_literal_expr_fn=_scalar_literal_expr,
        can_omit_materialized_channel_last_alias_fn=_can_omit_materialized_channel_last_alias,
        channel_first_binary_input_expr_fn=_channel_first_binary_input_expr,
        op=op,
    )

    producer_by_output_name: Dict[str, OperatorIR] = {}
    for operator in model_ir.operators:
        for output_name in operator.outputs:
            producer_by_output_name[str(output_name)] = operator

    _shape_tensor_length = lambda tensor_name: _shape_tensor_length_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )
    _reconstruct_shape_scalar_expr = lambda tensor_name, seen=None: _reconstruct_shape_scalar_expr_for_codegen(
        model_ir=model_ir,
        producer_by_output_name=producer_by_output_name,
        tensor_exact_static_shape_list_fn=_tensor_exact_static_shape_list,
        tensor_expr_fn=_tensor_expr,
        runtime_imports=runtime_imports,
        tensor_name=tensor_name,
        seen=seen,
    )
    _reconstruct_shape_list_expr = lambda tensor_name, seen=None: _reconstruct_shape_list_expr_for_codegen(
        model_ir=model_ir,
        producer_by_output_name=producer_by_output_name,
        tensor_exact_static_shape_list_fn=_tensor_exact_static_shape_list,
        tensor_expr_fn=_tensor_expr,
        runtime_imports=runtime_imports,
        tensor_name=tensor_name,
        seen=seen,
    )

    _match_if_axis0_tensor_mux_slice = lambda op: _match_if_axis0_tensor_mux_slice_for_codegen(
        model_ir=model_ir,
        producer_by_output_name=producer_by_output_name,
        op=op,
    )
    _binary_operand_expr = lambda tensor_name, other_tensor_name: _binary_operand_expr_for_codegen(
        model_ir=model_ir,
        binary_constant_buffer_alias_exprs=binary_constant_buffer_alias_exprs,
        channel_first_constant_buffer_alias_exprs=channel_first_constant_buffer_alias_exprs,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        tensor_expr_fn=_tensor_expr,
        channel_first_rank4_constant_buffer_alias_shape_fn=_channel_first_rank4_constant_buffer_alias_shape,
        channel_first_shape_for_tensor_fn=_channel_first_shape_for_tensor,
        constant_permute_for_broadcast_fn=_constant_permute_for_broadcast,
        permuted_constant_expr_for_tensor_name_fn=_permuted_constant_expr_for_tensor_name,
        tensor_name=tensor_name,
        other_tensor_name=other_tensor_name,
    )

    runtime_shape_uncertain_tensors: Set[str] = set()

    _binary_runtime_shape_passthrough_operand = lambda lhs_name, rhs_name: _binary_runtime_shape_passthrough_operand_for_codegen(
        model_ir=model_ir,
        runtime_shape_uncertain_tensors=runtime_shape_uncertain_tensors,
        lhs_name=lhs_name,
        rhs_name=rhs_name,
    )
    _binary_requires_runtime_alignment = lambda lhs_name, rhs_name, output_name: _binary_requires_runtime_alignment_for_codegen(
        model_ir=model_ir,
        runtime_shape_uncertain_tensors=runtime_shape_uncertain_tensors,
        lhs_name=lhs_name,
        rhs_name=rhs_name,
        output_name=output_name,
    )

    if _is_sequential_single_input_graph():
        tensor_var_names[str(model_ir.inputs[0])] = "x"
        for op in model_ir.operators:
            if str(op.op_type) == "SHAPE":
                continue
            tensor_var_names[str(op.outputs[0])] = "x"
    _has_channel_last_consumer_hint_for_same_shape_transpose = lambda op: _has_channel_last_consumer_hint_for_same_shape_transpose_for_codegen(
        model_ir=model_ir,
        consumer_index=consumer_index,
        op=op,
    )
    _is_batchless_rank3_public_output_transpose = lambda op: _is_batchless_rank3_public_output_transpose_for_codegen(
        model_ir=model_ir,
        producer_index=producer_index,
        batchless_rank3_public_boundary_names=batchless_rank3_public_boundary_names,
        op=op,
    )

    runtime_imports = context.runtime_imports
    runtime_imports.update({
        "resolve_named_input_value",
        "load_generated_weights",
    })
    batchless_rank3_public_boundary_names = {
        str(name)
        for name in list(model_ir.metadata.get("batchless_rank3_public_boundary_names", []))
    }
    forward_lines = context.forward_lines
    for op_index, op in enumerate(model_ir.operators):
        if int(op_index) in skipped_op_indices:
            continue
        op_type = str(op.op_type)
        outputs = [str(v) for v in op.outputs]
        output_vars = [tensor_var_names[str(name)] for name in outputs]
        output_target_shape = _target_shape_literal(outputs[0]) if len(outputs) == 1 else "None"
        if (
            op_type in {"GATHER", "GATHER_ND", "RESHAPE", "SLICE", "STRIDED_SLICE", "CONCATENATION"}
            and any(str(input_name) in runtime_shape_uncertain_tensors for input_name in op.inputs)
        ):
            runtime_shape_uncertain_tensors.update(outputs)
        affine_layer_norm_spec = affine_layer_norm_specs.get(int(op_index), None)
        if affine_layer_norm_spec is not None:
            forward_lines.append(
                f"{output_vars[0]} = self.{affine_layer_norm_spec['attr_name']}({_tensor_expr(str(affine_layer_norm_spec['input_name']))})"
            )
            continue
        if _emit_native_direct_module_op_for_codegen(
            model_ir=model_ir,
            op=op,
            op_index=int(op_index),
            outputs=outputs,
            output_vars=output_vars,
            output_target_shape=output_target_shape,
            op_module_attr_names=op_module_attr_names,
            fused_module_specs=fused_module_specs,
            conv_module_pad_specs=conv_module_pad_specs,
            tensor_var_names=tensor_var_names,
            channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            tensor_expr_fn=_tensor_expr,
            tensor_expr_for_channel_first_bridge_fn=_tensor_expr_for_channel_first_bridge,
            all_consumers_are_channel_first_binary_ops_fn=_all_consumers_are_channel_first_binary_ops,
            can_omit_materialized_channel_last_alias_fn=_can_omit_materialized_channel_last_alias,
            derived_local_var_name_fn=_derived_local_var_name,
            emit_module_output_expr_fn=_emit_module_output_expr,
            target_shape_literal_fn=_target_shape_literal,
            conv2d_input_pre_permute_fn=_conv2d_input_pre_permute,
            can_emit_direct_module_call_fn=_can_emit_direct_module_call,
            activation_lines_fn=_activation_lines,
            emit_maybe_aligned_expr_fn=_emit_maybe_aligned_expr,
            tensor_shape_list_fn=_tensor_shape_list,
            should_skip_align_for_shape_preserving_unary_fn=_should_skip_align_for_shape_preserving_unary,
        ):
            continue
        if _emit_native_binary_op_for_codegen(
            model_ir=model_ir,
            op=op,
            op_index=int(op_index),
            outputs=outputs,
            output_vars=output_vars,
            output_target_shape=output_target_shape,
            channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            runtime_shape_uncertain_tensors=runtime_shape_uncertain_tensors,
            tensor_dtype_name_fn=_tensor_dtype_name,
            binary_operand_expr_fn=_binary_operand_expr,
            scalar_literal_expr_fn=_scalar_literal_expr,
            can_emit_channel_first_binary_op_fn=_can_emit_channel_first_binary_op,
            channel_first_binary_input_expr_fn=_channel_first_binary_input_expr,
            derived_local_var_name_fn=_derived_local_var_name,
            can_omit_materialized_channel_last_alias_fn=_can_omit_materialized_channel_last_alias,
            target_shape_literal_fn=_target_shape_literal,
            emit_maybe_aligned_expr_fn=_emit_maybe_aligned_expr,
            binary_runtime_shape_passthrough_operand_fn=_binary_runtime_shape_passthrough_operand,
            binary_requires_runtime_alignment_fn=_binary_requires_runtime_alignment,
            preferred_binary_alignment_anchor_fn=lambda lhs_name, rhs_name, output_name: _preferred_binary_alignment_anchor_for_codegen(
                model_ir=model_ir,
                lhs_name=lhs_name,
                rhs_name=rhs_name,
                output_name=output_name,
            ),
            activation_lines_fn=_activation_lines,
        ):
            continue
        if _emit_native_unary_op_for_codegen(
            model_ir=model_ir,
            op=op,
            outputs=outputs,
            output_vars=output_vars,
            channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            tensor_expr_fn=_tensor_expr,
            channel_first_passthrough_input_expr_fn=_channel_first_passthrough_input_expr,
            can_emit_channel_first_shape_preserving_unary_op_fn=_can_emit_channel_first_shape_preserving_unary_op,
            derived_local_var_name_fn=_derived_local_var_name,
            can_omit_materialized_channel_last_alias_fn=_can_omit_materialized_channel_last_alias,
            target_shape_literal_fn=_target_shape_literal,
            tensor_shape_list_fn=_tensor_shape_list,
            should_skip_align_for_shape_preserving_unary_fn=_should_skip_align_for_shape_preserving_unary,
            emit_maybe_aligned_expr_fn=_emit_maybe_aligned_expr,
        ):
            continue
        if op_type == "GATHER":
            axis = int(op.options.get("axis", 0))
            batch_dims = int(op.options.get("batchDims", 0))
            params_expr = _tensor_expr(str(op.inputs[0]))
            gather_output_tensor = model_ir.tensors.get(str(outputs[0]), None)
            gather_single_consumer_indices = consumer_index.get(str(outputs[0]), [])
            if (
                gather_output_tensor is not None
                and len(gather_single_consumer_indices) == 1
            ):
                gather_consumer = model_ir.operators[int(gather_single_consumer_indices[0])]
                gather_consumer_output_tensor = (
                    model_ir.tensors.get(str(gather_consumer.outputs[0]), None)
                    if len(gather_consumer.outputs) == 1
                    else None
                )
                if (
                    str(gather_consumer.op_type) == "RESHAPE"
                    and gather_consumer_output_tensor is not None
                    and _is_suffix_flatten_gather_reshape(
                        gather_output_tensor.shape,
                        gather_consumer_output_tensor.shape,
                    )
                ):
                    continue
            if _should_elide_crd_to_dcr_gather_for_depth_to_space(
                model_ir=model_ir,
                params_name=str(op.inputs[0]),
                indices_name=str(op.inputs[1]),
                output_name=str(outputs[0]),
                axis=axis,
                batch_dims=batch_dims,
            ):
                inferred_shape = _tensor_shape_list(str(op.inputs[0]))
                if _should_skip_align_for_shape_preserving_unary(str(op.inputs[0]), outputs[0]):
                    forward_lines.append(f"{output_vars[0]} = {params_expr}")
                else:
                    forward_lines.append(
                        f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=params_expr, inferred_shape=inferred_shape)}"
                    )
                continue
            gather_pre_perm = _gather_input_pre_permute(
                params_name=str(op.inputs[0]),
                output_name=str(outputs[0]),
                axis=axis,
                batch_dims=batch_dims,
            )
            gather_input_rank = len(model_ir.tensors[str(op.inputs[0])].shape)
            if gather_pre_perm is not None:
                params_expr = f"{params_expr}.permute({', '.join(str(int(v)) for v in gather_pre_perm)}).contiguous()"
                gather_input_rank = int(len(gather_pre_perm))
            indices_expr = _tensor_expr(str(op.inputs[1]))
            indices_name = str(op.inputs[1])
            if _is_identity_nms_postprocess_gather(str(op.inputs[0]), indices_name):
                inferred_shape = _tensor_shape_list(str(op.inputs[0]))
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=params_expr, inferred_shape=inferred_shape)}"
                )
                continue
            direct_indices_values = _constant_int_list(model_ir.tensors.get(indices_name, None))
            direct_gather_expr = None
            indices_tensor = model_ir.tensors.get(indices_name, None)
            if direct_indices_values is not None:
                direct_gather_expr = _direct_gather_expr(
                    params_expr=params_expr,
                    indices_values=direct_indices_values,
                    indices_shape=indices_tensor.shape if indices_tensor is not None else None,
                    axis=axis,
                    batch_dims=batch_dims,
                    input_rank=gather_input_rank,
                )
            if direct_gather_expr is None:
                direct_gather_expr = _direct_dynamic_gather_expr(
                    params_expr=params_expr,
                    indices_expr=indices_expr,
                    axis=axis,
                    batch_dims=batch_dims,
                    input_rank=gather_input_rank,
                    indices_name=indices_name,
                    indices_shape=(
                        indices_tensor.shape
                        if indices_tensor is not None
                        else None
                    ),
                    indices_shape_signature=(
                        indices_tensor.shape_signature
                        if indices_tensor is not None
                        else None
                    ),
                )
            if direct_gather_expr is not None:
                if "_reshape_gather_output(" in direct_gather_expr:
                    runtime_imports.add("_reshape_gather_output")
                if "_shape_tensor(" in direct_gather_expr:
                    runtime_imports.add("_shape_tensor")
                forward_lines.append(f"{output_vars[0]} = {direct_gather_expr}")
            else:
                runtime_imports.add("_apply_gather")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_gather({params_expr}, {indices_expr}, axis={axis}, batch_dims={batch_dims}, target_shape={output_target_shape}, indices_name={indices_name!r})"
                )
            continue
        if op_type == "GATHER_ND":
            params_name = str(op.inputs[0])
            params_expr = _tensor_expr(params_name)
            indices_expr = _tensor_expr(str(op.inputs[1]))
            params_tensor = model_ir.tensors.get(params_name, None)
            public_layout_map = cast(
                Dict[str, str],
                model_ir.metadata.get("onnx_public_layout_map", {}),
            )
            if params_tensor is not None and params_name in model_ir.inputs:
                params_layout = normalize_logical_layout(params_tensor.logical_layout)
                public_layout = normalize_logical_layout(
                    public_layout_map.get(params_name, LOGICAL_LAYOUT_UNKNOWN)
                )
                if (
                    params_layout != LOGICAL_LAYOUT_UNKNOWN
                    and public_layout != LOGICAL_LAYOUT_UNKNOWN
                    and params_layout != public_layout
                ):
                    boundary_perm = logical_layout_permutation(
                        source_layout=public_layout,
                        target_layout=params_layout,
                    )
                    if boundary_perm is not None:
                        runtime_imports.add("_torch_permute")
                        params_expr = f"_torch_permute({params_expr}, {repr([int(v) for v in list(boundary_perm)])})"
            if params_tensor is not None:
                params_layout = normalize_logical_layout(params_tensor.logical_layout)
                params_actual_shape = [int(v) for v in list(params_tensor.shape)]
                params_semantic_shape = _tensor_shape_list(params_name)
                params_rank = len(params_actual_shape)
                if (
                    params_semantic_shape is not None
                    and params_rank in {3, 4, 5}
                    and is_channel_first_logical_layout(params_layout)
                ):
                    cf_to_cl_perm = logical_layout_permutation(
                        source_layout=params_layout,
                        target_layout=channel_last_logical_layout(params_rank),
                    )
                    if (
                        cf_to_cl_perm is not None
                        and _shape_lists_equal(
                            _permute_shape(params_actual_shape, cf_to_cl_perm),
                            params_semantic_shape,
                        )
                    ):
                        runtime_imports.add("_torch_permute")
                        params_expr = f"_torch_permute({params_expr}, {repr([int(v) for v in list(cf_to_cl_perm)])})"
                output_tensor_for_gather_nd = model_ir.tensors.get(str(outputs[0]), None)
                output_shape_for_gather_nd = (
                    [int(v) for v in list(output_tensor_for_gather_nd.shape)]
                    if output_tensor_for_gather_nd is not None
                    else _tensor_shape_list(str(outputs[0]))
                )
                if (
                    output_shape_for_gather_nd is not None
                    and params_rank in {3, 4, 5}
                    and is_channel_first_logical_layout(params_layout)
                ):
                    cf_to_cl_perm = logical_layout_permutation(
                        source_layout=params_layout,
                        target_layout=channel_last_logical_layout(params_rank),
                    )
                    if cf_to_cl_perm is not None:
                        actual_gather_nd_shape = _infer_gather_nd_shape(
                            params_shape=params_actual_shape,
                            indices_tensor_name=str(op.inputs[1]),
                        )
                        permuted_actual_shape = _permute_shape(params_actual_shape, cf_to_cl_perm)
                        permuted_gather_nd_shape = _infer_gather_nd_shape(
                            params_shape=permuted_actual_shape,
                            indices_tensor_name=str(op.inputs[1]),
                        )
                        if (
                            not _shape_lists_equal(actual_gather_nd_shape, output_shape_for_gather_nd)
                            and _shape_lists_equal(permuted_gather_nd_shape, output_shape_for_gather_nd)
                        ):
                            runtime_imports.add("_torch_permute")
                            params_expr = f"_torch_permute({params_expr}, {repr([int(v) for v in list(cf_to_cl_perm)])})"
            gather_elements_axis_hint = None
            gather_elements_indices_name = None
            if str(op.inputs[1]).endswith("_gather_elements_coords"):
                producer_op_idx = producer_index.get(str(op.inputs[1]), None)
                if producer_op_idx is not None:
                    producer_op = model_ir.operators[int(producer_op_idx)]
                    if str(producer_op.op_type) == "CONCATENATION":
                        gather_elements_axis_hint = next(
                            (
                                idx for idx, name in enumerate(producer_op.inputs)
                                if str(name).endswith("_gather_elements_axis_coord")
                            ),
                            None,
                        )
                candidate_indices_name = str(op.inputs[1]).replace("_gather_elements_coords", "_gather_elements_indices_i32")
                if candidate_indices_name in model_ir.tensors:
                    gather_elements_indices_name = candidate_indices_name
            if gather_elements_axis_hint is not None and gather_elements_indices_name is not None:
                params_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                params_rank = 0 if params_tensor is None else len(list(params_tensor.shape))
                gather_axis = int(gather_elements_axis_hint)
                gather_axis_cl = None
                params_layout = (
                    LOGICAL_LAYOUT_UNKNOWN
                    if params_tensor is None
                    else normalize_logical_layout(params_tensor.logical_layout)
                )
                if params_rank in {3, 4, 5} and is_channel_first_logical_layout(params_layout):
                    perm_cf_to_cl = logical_layout_permutation(
                        source_layout=channel_first_logical_layout(params_rank),
                        target_layout=channel_last_logical_layout(params_rank),
                    )
                    if perm_cf_to_cl is not None and gather_axis in perm_cf_to_cl:
                        gather_axis_cl = int(perm_cf_to_cl.index(gather_axis))
                runtime_imports.add("_apply_gather_elements")
                runtime_shape_uncertain_tensors.add(outputs[0])
                forward_lines.append(
                    f"{output_vars[0]} = _apply_gather_elements({params_expr}, {_tensor_expr(gather_elements_indices_name)}, axis_hint={gather_axis}, axis_hint_cl={repr(gather_axis_cl)}, target_shape={output_target_shape})"
                )
                continue
            params_shape = (
                [int(v) for v in list(params_tensor.shape)]
                if params_tensor is not None and params_name in model_ir.inputs
                else _tensor_shape_list(str(op.inputs[0]))
            )
            output_shape = _tensor_shape_list(outputs[0])
            if (
                params_tensor is not None
                and params_shape is not None
                and output_shape is not None
                and len(list(params_shape)) in {3, 4, 5}
            ):
                params_layout = normalize_logical_layout(params_tensor.logical_layout)
                expected_shape_from_params = _infer_gather_nd_shape(params_shape, str(op.inputs[1]))
                if not _shape_lists_equal(expected_shape_from_params, output_shape):
                    alternate_layout = None
                    rank = len(list(params_shape))
                    if is_channel_first_logical_layout(params_layout):
                        alternate_layout = channel_last_logical_layout(rank)
                    elif is_channel_last_logical_layout(params_layout):
                        alternate_layout = channel_first_logical_layout(rank)
                    if alternate_layout is not None:
                        perm = logical_layout_permutation(
                            source_layout=params_layout,
                            target_layout=alternate_layout,
                        )
                        if perm is not None:
                            resolved_perm = perm
                            permuted_params_shape = _permute_shape(params_shape, resolved_perm)
                            if (
                                permuted_params_shape is not None
                                and _shape_lists_equal(
                                    _infer_gather_nd_shape(permuted_params_shape, str(op.inputs[1])),
                                    output_shape,
                                )
                            ):
                                params_expr = f"{params_expr}.permute({', '.join(str(int(v)) for v in resolved_perm)}).contiguous()"
            runtime_imports.add("_apply_gather_nd")
            forward_lines.append(
                f"{output_vars[0]} = _apply_gather_nd({params_expr}, {indices_expr}, target_shape={output_target_shape})"
            )
            continue
        if op_type == "CAST":
            out_dtype = str(op.options.get("outDataType", "FLOAT32"))
            forward_lines.append(
                f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}.to(dtype={_torch_dtype_literal(out_dtype)})"
            )
            continue
        if op_type == "LOCAL_RESPONSE_NORMALIZATION":
            input_expr = _tensor_expr(str(op.inputs[0]))
            radius = int(op.options.get("radius", 0))
            size = max(1, int(radius) * 2 + 1)
            alpha = float(op.options.get("alpha", 1.0))
            beta = float(op.options.get("beta", 0.5))
            bias = float(op.options.get("bias", 1.0))
            inferred_shape = _tensor_shape_list(str(op.inputs[0]))
            forward_lines.append(
                f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'F.local_response_norm({input_expr}, size={size}, alpha={alpha}, beta={beta}, k={bias})', inferred_shape=inferred_shape)}"
            )
            continue
        if op_type == "RESHAPE":
            runtime_imports.add("_shape_list")
            reshape_input_expr = _tensor_expr(str(op.inputs[0]))
            reshape_input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            reshape_output_tensor = model_ir.tensors.get(str(outputs[0]), None)
            reshape_is_lowered_onnx_flatten = "onnxFlattenAxis" in op.options
            reshape_allow_zero = bool(op.options.get("allowZero", False))
            reshape_input_shape = None if reshape_input_tensor is None else [int(v) for v in list(reshape_input_tensor.shape)]
            reshape_output_shape = None if reshape_output_tensor is None else [int(v) for v in list(reshape_output_tensor.shape)]
            reshape_output_preferred_shape = _preferred_reshape_target_values(reshape_output_tensor)
            reshape_output_semantic_shape = (
                [int(v) for v in list(reshape_output_preferred_shape)]
                if reshape_output_preferred_shape is not None
                else reshape_output_shape
            )
            reshape_input_layout = None if reshape_input_tensor is None else str(reshape_input_tensor.logical_layout)
            reshape_output_layout = None if reshape_output_tensor is None else str(reshape_output_tensor.logical_layout)
            if (
                not reshape_is_lowered_onnx_flatten
                and reshape_input_shape is not None
                and len(reshape_input_shape) == 4
                and reshape_output_semantic_shape is not None
                and len(reshape_output_semantic_shape) in {2, 3}
            ):
                effective_layout = _infer_effective_rank4_runtime_layout(str(op.inputs[0]))
                if effective_layout is not None:
                    reshape_input_layout = effective_layout
            reshape_special_plan = None
            reshape_pre_perm = None
            reshape_feature_last_target = None
            reshape_channel_first_alias_shape = None
            gather_flatten_direct_spec: Optional[Dict[str, Any]] = None
            reshape_plain_singleton_axis_drop = _reshape_is_plain_singleton_axis_drop(
                reshape_input_shape,
                reshape_output_semantic_shape,
            )
            reshape_input_producer_idx = producer_index.get(str(op.inputs[0]), None)
            if reshape_input_producer_idx is not None:
                reshape_input_producer = model_ir.operators[int(reshape_input_producer_idx)]
                if (
                    str(reshape_input_producer.op_type) == "GATHER"
                    and len(consumer_index.get(str(op.inputs[0]), [])) == 1
                    and _is_suffix_flatten_gather_reshape(
                        reshape_input_shape,
                        reshape_output_semantic_shape,
                    )
                ):
                    gather_axis = int(reshape_input_producer.options.get("axis", 0))
                    gather_batch_dims = int(reshape_input_producer.options.get("batchDims", 0))
                    gather_params_name = str(reshape_input_producer.inputs[0])
                    gather_indices_name = str(reshape_input_producer.inputs[1])
                    gather_params_expr = _tensor_expr(gather_params_name)
                    gather_input_rank = len(model_ir.tensors[gather_params_name].shape)
                    gather_pre_perm = _gather_input_pre_permute(
                        params_name=gather_params_name,
                        output_name=str(op.inputs[0]),
                        axis=gather_axis,
                        batch_dims=gather_batch_dims,
                    )
                    if gather_pre_perm is not None:
                        gather_params_expr = (
                            f"{gather_params_expr}.permute("
                            f"{', '.join(str(int(v)) for v in gather_pre_perm)}).contiguous()"
                        )
                        gather_input_rank = int(len(gather_pre_perm))
                    gather_flatten_direct_spec = {
                        "params_expr": gather_params_expr,
                        "indices_expr": _tensor_expr(gather_indices_name),
                        "indices_values": _constant_int_list(model_ir.tensors.get(gather_indices_name, None)),
                        "indices_shape": (
                            model_ir.tensors[gather_indices_name].shape
                            if gather_indices_name in model_ir.tensors
                            else None
                        ),
                        "indices_shape_signature": (
                            model_ir.tensors[gather_indices_name].shape_signature
                            if gather_indices_name in model_ir.tensors
                            else None
                        ),
                        "axis": gather_axis,
                        "batch_dims": gather_batch_dims,
                        "input_rank": gather_input_rank,
                        "indices_name": gather_indices_name,
                    }
            if not reshape_is_lowered_onnx_flatten and not reshape_plain_singleton_axis_drop:
                reshape_special_plan = _reshape_special_layout_plan(
                    input_shape=reshape_input_shape,
                    output_shape=reshape_output_semantic_shape,
                    input_layout=reshape_input_layout,
                    output_layout=reshape_output_layout,
                )
                reshape_pre_perm = _reshape_preserves_channel_last_sequence(
                    reshape_input_shape,
                    reshape_output_semantic_shape,
                    reshape_input_layout,
                )
                if reshape_special_plan is not None and reshape_special_plan.get("pre_perm", None) is not None:
                    reshape_pre_perm = list(reshape_special_plan["pre_perm"])
                reshape_feature_last_target = _reshape_prefers_feature_last_for_adjx_batch_matmul(
                    str(op.inputs[0]),
                    str(outputs[0]),
                )
                if reshape_feature_last_target is not None:
                    reshape_pre_perm = list(reshape_feature_last_target[0])
            if (
                reshape_pre_perm is None
                and reshape_special_plan is None
                and reshape_feature_last_target is None
                and reshape_input_shape is not None
                and reshape_output_shape is not None
                and len(reshape_input_shape) == 2
                and len(reshape_output_shape) == 4
                and reshape_output_tensor is not None
                and is_channel_last_logical_layout(
                    normalize_logical_layout(reshape_output_tensor.logical_layout)
                )
                and all(int(dim) == 1 for dim in list(reshape_output_shape[1:-1]))
                and _shape_lists_equal_relaxed(
                    reshape_input_shape,
                    [int(reshape_output_shape[0]), int(reshape_output_shape[-1])],
                )
            ):
                reshape_channel_first_alias_shape = [
                    int(reshape_output_shape[0]),
                    int(reshape_output_shape[-1]),
                    *[int(v) for v in list(reshape_output_shape[1:-1])],
                ]
            collapsed_reshape_shape_values: Optional[List[int]] = None
            collapsed_reshape_input_expr: Optional[str] = None
            upstream_reshape_index = producer_index.get(str(op.inputs[0]), None)
            if (
                upstream_reshape_index is not None
                and len(consumer_index.get(str(op.inputs[0]), [])) == 1
                and reshape_pre_perm is None
                and reshape_special_plan is None
                and reshape_feature_last_target is None
                and reshape_channel_first_alias_shape is None
                and gather_flatten_direct_spec is None
            ):
                upstream_reshape_op = model_ir.operators[int(upstream_reshape_index)]
                candidate_shape_values = (
                    _preferred_reshape_target_values(reshape_output_tensor)
                    if reshape_output_tensor is not None
                    else None
                )
                if candidate_shape_values is None and reshape_output_shape is not None:
                    candidate_shape_values = [int(v) for v in list(reshape_output_shape)]
                if (
                    candidate_shape_values is not None
                    and len(candidate_shape_values) > 0
                    and all(int(v) > 0 for v in list(candidate_shape_values))
                    and _can_emit_direct_torch_reshape_shape(candidate_shape_values, allow_zero=reshape_allow_zero)
                    and _reshape_codegen_is_plain_data_only(upstream_reshape_op)
                ):
                    collapsed_reshape_shape_values = [int(v) for v in list(candidate_shape_values)]
                    collapsed_reshape_input_expr = _tensor_expr(str(upstream_reshape_op.inputs[0]))
            if collapsed_reshape_shape_values is not None and collapsed_reshape_input_expr is not None:
                channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                forward_lines.append(
                    f"{output_vars[0]} = torch.reshape({collapsed_reshape_input_expr}, {repr(collapsed_reshape_shape_values)})"
                )
                continue
            if reshape_pre_perm is not None:
                reshape_input_expr = f"{reshape_input_expr}.permute({', '.join(str(int(v)) for v in reshape_pre_perm)}).contiguous()"
            shape_is_tensor_expr = False
            if reshape_feature_last_target is not None:
                feature_last_shape_values = [int(v) for v in list(reshape_feature_last_target[1])]
                shape_expr = f"[int(v) for v in {repr(feature_last_shape_values)}]"
            elif reshape_special_plan is not None and reshape_special_plan.get("reshape_shape", None) is not None:
                shape_expr = repr([int(v) for v in list(reshape_special_plan["reshape_shape"])])
            elif len(op.inputs) >= 2:
                const_shape_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
                if const_shape_values is not None:
                    direct_shape_values = [int(v) for v in list(const_shape_values)]
                    preserve_feature_last_rank3_shape_expr = bool(
                        reshape_output_tensor is not None
                        and reshape_input_shape is not None
                        and len(reshape_input_shape) == 2
                        and len(list(reshape_output_tensor.shape)) == 3
                        and is_channel_last_logical_layout(
                            normalize_logical_layout(reshape_output_tensor.logical_layout)
                        )
                    )
                    if _can_emit_direct_torch_reshape_shape(
                        direct_shape_values,
                        allow_zero=reshape_allow_zero,
                    ) and all(int(v) > 0 for v in direct_shape_values):
                        if preserve_feature_last_rank3_shape_expr:
                            shape_expr = f"[int(v) for v in {repr(direct_shape_values)}]"
                        else:
                            shape_expr = repr(direct_shape_values)
                    else:
                        runtime_imports.add("_resolve_reshape_shape")
                        shape_expr = (
                            f"_resolve_reshape_shape({repr(direct_shape_values)}, "
                            f"{reshape_input_expr}, allow_zero={reshape_allow_zero})"
                        )
                else:
                    reconstructed_shape_expr = _reconstruct_shape_list_expr(str(op.inputs[1]))
                    if reconstructed_shape_expr is not None:
                        shape_expr = reconstructed_shape_expr
                    else:
                        shape_is_tensor_expr = True
                        if (
                            reshape_allow_zero
                            or _reshape_shape_tensor_uses_runtime_dims(str(op.inputs[1]))
                        ):
                            runtime_imports.add("_shape_list")
                            shape_expr = f"{_tensor_expr(str(op.inputs[1]))}.to(dtype=torch.int64).reshape(-1)"
                        else:
                            runtime_imports.add("_resolve_reshape_shape_tensor")
                            shape_expr = (
                                f"_resolve_reshape_shape_tensor({_tensor_expr(str(op.inputs[1]))}, "
                                f"{reshape_input_expr}, allow_zero=False)"
                            )
            elif reshape_output_preferred_shape is not None:
                preferred_shape_values = [int(v) for v in list(reshape_output_preferred_shape)]
                if _can_emit_direct_torch_reshape_shape(
                    preferred_shape_values,
                    allow_zero=reshape_allow_zero,
                ) and all(int(v) > 0 for v in preferred_shape_values):
                    shape_expr = repr(preferred_shape_values)
                else:
                    runtime_imports.add("_resolve_reshape_shape")
                    shape_expr = (
                        f"_resolve_reshape_shape({repr(preferred_shape_values)}, "
                        f"{reshape_input_expr}, allow_zero={reshape_allow_zero})"
                    )
            else:
                runtime_imports.add("_resolve_reshape_shape")
                raw_new_shape = op.options.get("onnxRawNewShape", op.options.get("newShape", []))
                shape_expr = (
                    f"_resolve_reshape_shape({repr([int(v) for v in list(raw_new_shape)])}, "
                    f"{reshape_input_expr}, allow_zero={reshape_allow_zero})"
                )
            raw_new_shape = op.options.get("onnxRawNewShape", op.options.get("newShape", None))
            reshape_is_gather_elements_axis_coord = bool(
                str(op.inputs[0]).endswith("_gather_elements_indices_i32")
                and str(outputs[0]).endswith("_gather_elements_axis_coord")
                and reshape_input_shape is not None
                and isinstance(raw_new_shape, list)
                and [int(v) for v in list(raw_new_shape)] == [*reshape_input_shape, 1]
            )
            if reshape_is_gather_elements_axis_coord:
                forward_lines.append(f"{output_vars[0]} = torch.unsqueeze({reshape_input_expr}, dim=-1)")
            else:
                reshape_shape_arg = f"_shape_list({shape_expr})" if shape_is_tensor_expr else shape_expr
                direct_gather_reshape_expr = None
                if (
                    gather_flatten_direct_spec is not None
                    and reshape_pre_perm is None
                    and reshape_special_plan is None
                    and reshape_feature_last_target is None
                    and reshape_channel_first_alias_shape is None
                ):
                    direct_gather_reshape_expr = _direct_gather_reshape_expr(
                        params_expr=str(gather_flatten_direct_spec["params_expr"]),
                        indices_expr=str(gather_flatten_direct_spec["indices_expr"]),
                        indices_values=cast(Optional[Sequence[int]], gather_flatten_direct_spec["indices_values"]),
                        indices_shape=cast(Optional[Sequence[int]], gather_flatten_direct_spec["indices_shape"]),
                        indices_shape_signature=cast(Optional[Sequence[int]], gather_flatten_direct_spec["indices_shape_signature"]),
                        axis=int(gather_flatten_direct_spec["axis"]),
                        batch_dims=int(gather_flatten_direct_spec["batch_dims"]),
                        input_rank=int(gather_flatten_direct_spec["input_rank"]),
                        indices_name=str(gather_flatten_direct_spec["indices_name"]),
                        final_shape_values=(
                            reshape_output_preferred_shape
                            if reshape_output_preferred_shape is not None
                            else reshape_output_shape
                        ),
                    )
                if reshape_channel_first_alias_shape is not None:
                    output_name = str(outputs[0])
                    raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                    channel_first_tensor_expr_aliases[output_name] = raw_output_var
                    forward_lines.append(
                        f"{raw_output_var} = torch.reshape({reshape_input_expr}, {repr(reshape_channel_first_alias_shape)})"
                    )
                    if _can_omit_materialized_channel_last_alias(output_name):
                        continue
                    runtime_imports.add("_align_tensor_to_target_shape")
                    forward_lines.append(
                        f"{output_vars[0]} = _align_tensor_to_target_shape("
                        f"{raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                        f"{_target_shape_literal(output_name)})"
                    )
                else:
                    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                    forward_lines.append(
                        f"{output_vars[0]} = {direct_gather_reshape_expr}"
                        if direct_gather_reshape_expr is not None
                        else f"{output_vars[0]} = torch.reshape({reshape_input_expr}, {reshape_shape_arg})"
                    )
            if reshape_special_plan is not None and reshape_special_plan.get("post_perm", None) is not None:
                post_perm = [int(v) for v in list(reshape_special_plan["post_perm"])]
                forward_lines.append(
                    f"{output_vars[0]} = {output_vars[0]}.permute({', '.join(str(v) for v in post_perm)}).contiguous()"
                )
            continue
        if _emit_native_transpose_op_for_codegen(
            model_ir=model_ir,
            op=op,
            outputs=outputs,
            output_vars=output_vars,
            preserve_channel_last_tensor_names=preserve_channel_last_tensor_names,
            consumer_index=consumer_index,
            producer_index=producer_index,
            channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            tensor_expr_fn=_tensor_expr,
            tensor_expr_for_channel_first_bridge_fn=_tensor_expr_for_channel_first_bridge,
            can_fold_channel_last_alias_slice_consumer_fn=_can_fold_channel_last_alias_slice_consumer,
            all_consumers_are_channel_first_binary_ops_fn=_all_consumers_are_channel_first_binary_ops,
            can_omit_materialized_channel_last_alias_fn=_can_omit_materialized_channel_last_alias,
            has_channel_last_consumer_hint_for_same_shape_transpose_fn=_has_channel_last_consumer_hint_for_same_shape_transpose,
            is_batchless_rank3_public_output_transpose_fn=_is_batchless_rank3_public_output_transpose,
            target_shape_literal_fn=_target_shape_literal,
        ):
            continue
        if _emit_native_shape_transform_misc_op_for_codegen(
            model_ir=model_ir,
            op=op,
            op_type=op_type,
            outputs=outputs,
            output_vars=output_vars,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            tensor_expr_fn=_tensor_expr,
            axis_expr_from_input_fn=_axis_expr_from_input,
        ):
            continue
        if _emit_native_concat_op_for_codegen(
            model_ir=model_ir,
            op=op,
            op_index=op_index,
            outputs=outputs,
            output_vars=output_vars,
            output_target_shape=output_target_shape,
            channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            tensor_expr_fn=_tensor_expr,
            derived_local_var_name_fn=_derived_local_var_name,
            activation_lines_fn=_activation_lines,
            resolve_concat_axis_for_channel_first_fn=_resolve_concat_axis_for_channel_first,
            channel_first_concat_input_expr_fn=_channel_first_concat_input_expr,
            tensor_shape_list_fn=_tensor_shape_list,
            can_omit_materialized_channel_last_alias_fn=_can_omit_materialized_channel_last_alias,
            target_shape_literal_fn=_target_shape_literal,
            tensor_exact_static_shape_list_fn=_tensor_exact_static_shape_list,
            target_shape_values_fn=_target_shape_values,
        ):
            continue
        if op_type == "SLICE":
            slice_input_name = str(op.inputs[0])
            slice_input_tensor = model_ir.tensors.get(slice_input_name, None)
            slice_input_expr = _tensor_expr(slice_input_name)
            slice_begin_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None)) or []
            slice_size_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[2]), None)) or []
            slice_input_rank = len(model_ir.tensors[slice_input_name].shape)
            slice_input_shape = model_ir.tensors[slice_input_name].shape
            slice_output_tensor = model_ir.tensors.get(outputs[0], None)
            slice_folded_channel_first = False
            if (
                slice_input_tensor is not None
                and slice_input_name in channel_first_tensor_expr_aliases
                and is_channel_last_logical_layout(
                    normalize_logical_layout(slice_input_tensor.logical_layout)
                )
                and len(slice_begin_values) == slice_input_rank
                and len(slice_size_values) == slice_input_rank
            ):
                perm_from_cf = _perm_cf_to_cl(slice_input_rank)
                if perm_from_cf is not None:
                    slice_input_expr = str(channel_first_tensor_expr_aliases[slice_input_name])
                    slice_begin_values = _remap_axis_values_through_permutation(
                        slice_begin_values,
                        perm_from_cf,
                    )
                    slice_size_values = _remap_axis_values_through_permutation(
                        slice_size_values,
                        perm_from_cf,
                    )
                    perm_to_cf = _perm_cl_to_cf(slice_input_rank)
                    if perm_to_cf is not None:
                        slice_input_shape = _permute_shape(slice_input_shape, perm_to_cf) or slice_input_shape
                    slice_folded_channel_first = True
            direct_slice_expr = _direct_slice_expr(
                x_expr=slice_input_expr,
                begin_values=slice_begin_values,
                size_values=slice_size_values,
                input_rank=slice_input_rank,
                input_shape=slice_input_shape,
            )
            if direct_slice_expr is not None:
                if slice_folded_channel_first and slice_output_tensor is not None:
                    output_name = str(outputs[0])
                    output_rank = len(list(slice_output_tensor.shape))
                    output_layout = normalize_logical_layout(slice_output_tensor.logical_layout)
                    raw_output_var = (
                        output_vars[0]
                        if is_channel_first_logical_layout(output_layout)
                        else _derived_local_var_name(f"{output_vars[0]}_cf")
                    )
                    channel_first_tensor_expr_aliases[output_name] = raw_output_var
                    forward_lines.append(f"{raw_output_var} = {direct_slice_expr}")
                    if raw_output_var != output_vars[0]:
                        if _can_keep_channel_first_slice_output(output_name):
                            continue
                        perm_to_output = logical_layout_permutation(
                            source_layout=channel_first_logical_layout(output_rank),
                            target_layout=output_layout,
                        )
                        if perm_to_output is None:
                            raise ModelIRPyTorchExportError(
                                "Native PyTorch-like model.py codegen could not derive a folded slice layout bridge. "
                                f"output={output_name} output_layout={output_layout} rank={output_rank}"
                            )
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_vars[0]} = _align_tensor_to_target_shape("
                            f"{raw_output_var}.permute({', '.join(str(int(v)) for v in perm_to_output)}).contiguous(), "
                            f"{_target_shape_literal(output_name)})"
                        )
                    continue
                forward_lines.append(f"{output_vars[0]} = {direct_slice_expr}")
            else:
                if_axis0_tensor_mux_match = _match_if_axis0_tensor_mux_slice(op)
                if if_axis0_tensor_mux_match is not None:
                    runtime_imports.add("_apply_if_axis0_tensor_mux")
                    forward_lines.append(
                        f"{output_vars[0]} = _apply_if_axis0_tensor_mux("
                        f"{_tensor_expr(if_axis0_tensor_mux_match['cond_name'])}, "
                        f"{_tensor_expr(if_axis0_tensor_mux_match['then_name'])}, "
                        f"{_tensor_expr(if_axis0_tensor_mux_match['else_name'])}, "
                        f"{_tensor_expr(str(op.inputs[0]))}, "
                        f"{_tensor_expr(str(op.inputs[1]))}, "
                        f"{_tensor_expr(str(op.inputs[2]))}, "
                        f"target_shape={output_target_shape}, "
                        "use_export_mode=self._onnx2tf_torch_export_mode)"
                    )
                else:
                    runtime_imports.add("_apply_slice")
                    forward_lines.append(
                        f"{output_vars[0]} = _apply_slice({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))}, target_shape={output_target_shape})"
                    )
            continue
        if op_type == "STRIDED_SLICE":
            options = dict(op.options)
            strided_slice_input_name = str(op.inputs[0])
            strided_slice_input_tensor = model_ir.tensors.get(strided_slice_input_name, None)
            strided_slice_input_expr = _tensor_expr(strided_slice_input_name)
            strided_slice_begin_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None)) or []
            strided_slice_end_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[2]), None)) or []
            strided_slice_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[3]), None)) or []
            strided_slice_input_rank = len(model_ir.tensors[strided_slice_input_name].shape)
            strided_slice_begin_mask = int(options.get("beginMask", 0))
            strided_slice_end_mask = int(options.get("endMask", 0))
            strided_slice_folded_channel_first = False
            if (
                strided_slice_input_tensor is not None
                and strided_slice_input_name in channel_first_tensor_expr_aliases
                and is_channel_last_logical_layout(
                    normalize_logical_layout(strided_slice_input_tensor.logical_layout)
                )
                and len(strided_slice_begin_values) == strided_slice_input_rank
                and len(strided_slice_end_values) == strided_slice_input_rank
                and len(strided_slice_values) == strided_slice_input_rank
                and int(options.get("ellipsisMask", 0)) == 0
                and int(options.get("newAxisMask", 0)) == 0
                and int(options.get("shrinkAxisMask", 0)) == 0
            ):
                perm_from_cf = _perm_cf_to_cl(strided_slice_input_rank)
                if perm_from_cf is not None:
                    strided_slice_input_expr = str(channel_first_tensor_expr_aliases[strided_slice_input_name])
                    strided_slice_begin_values = _remap_axis_values_through_permutation(
                        strided_slice_begin_values,
                        perm_from_cf,
                    )
                    strided_slice_end_values = _remap_axis_values_through_permutation(
                        strided_slice_end_values,
                        perm_from_cf,
                    )
                    strided_slice_values = _remap_axis_values_through_permutation(
                        strided_slice_values,
                        perm_from_cf,
                    )
                    strided_slice_begin_mask = _remap_mask_bits_through_permutation(
                        strided_slice_begin_mask,
                        perm_from_cf,
                    )
                    strided_slice_end_mask = _remap_mask_bits_through_permutation(
                        strided_slice_end_mask,
                        perm_from_cf,
                    )
                    strided_slice_folded_channel_first = True
            direct_strided_slice_expr = _direct_strided_slice_expr(
                x_expr=strided_slice_input_expr,
                begin_values=strided_slice_begin_values,
                end_values=strided_slice_end_values,
                stride_values=strided_slice_values,
                begin_mask=strided_slice_begin_mask,
                end_mask=strided_slice_end_mask,
                input_rank=strided_slice_input_rank,
            )
            if direct_strided_slice_expr is None:
                direct_strided_slice_expr = _direct_symbolic_strided_slice_expr(
                    x_expr=strided_slice_input_expr,
                    begin_values=strided_slice_begin_values,
                    stride_values=strided_slice_values,
                    begin_mask=strided_slice_begin_mask,
                    end_mask=strided_slice_end_mask,
                    input_rank=strided_slice_input_rank,
                    end_list_expr=_reconstruct_shape_list_expr(str(op.inputs[2])),
                    end_scalar_expr=_reconstruct_shape_scalar_expr(str(op.inputs[2])),
                )
            if direct_strided_slice_expr is not None:
                strided_slice_output_tensor = model_ir.tensors.get(outputs[0], None)
                if strided_slice_folded_channel_first and strided_slice_output_tensor is not None:
                    output_name = str(outputs[0])
                    output_rank = len(list(strided_slice_output_tensor.shape))
                    output_layout = normalize_logical_layout(strided_slice_output_tensor.logical_layout)
                    raw_output_var = (
                        output_vars[0]
                        if is_channel_first_logical_layout(output_layout)
                        else _derived_local_var_name(f"{output_vars[0]}_cf")
                    )
                    channel_first_tensor_expr_aliases[output_name] = raw_output_var
                    forward_lines.append(f"{raw_output_var} = {direct_strided_slice_expr}")
                    if raw_output_var != output_vars[0]:
                        if _can_keep_channel_first_slice_output(output_name):
                            continue
                        perm_to_output = logical_layout_permutation(
                            source_layout=channel_first_logical_layout(output_rank),
                            target_layout=output_layout,
                        )
                        if perm_to_output is None:
                            raise ModelIRPyTorchExportError(
                                "Native PyTorch-like model.py codegen could not derive a folded strided-slice layout bridge. "
                                f"output={output_name} output_layout={output_layout} rank={output_rank}"
                            )
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_vars[0]} = _align_tensor_to_target_shape("
                            f"{raw_output_var}.permute({', '.join(str(int(v)) for v in perm_to_output)}).contiguous(), "
                            f"{_target_shape_literal(output_name)})"
                        )
                    continue
                forward_lines.append(f"{output_vars[0]} = {direct_strided_slice_expr}")
            else:
                runtime_imports.add("_apply_strided_slice")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_strided_slice({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))}, {_tensor_expr(str(op.inputs[3]))}, begin_mask={int(options.get('beginMask', 0))}, end_mask={int(options.get('endMask', 0))}, target_shape={output_target_shape})"
                )
            continue
        if op_type == "SHAPE":
            out_dtype = str(op.options.get("outType", "INT32"))
            runtime_imports.add("_shape_tensor")
            forward_lines.append(
                f"{output_vars[0]} = _shape_tensor({_tensor_expr(str(op.inputs[0]))}, dtype={_torch_dtype_literal(out_dtype)}, device={_tensor_expr(str(op.inputs[0]))}.device)"
            )
            continue
        if op_type == "FILL":
            fill_value_expr = _scalar_literal_expr(str(op.inputs[1])) or f"{_tensor_expr(str(op.inputs[1]))}.reshape(-1)[0].item()"
            fill_shape_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[0]), None))
            fill_shape_input_name = str(op.inputs[0])
            fill_shape_input_producer = producer_by_output_name.get(fill_shape_input_name, None)
            fill_value_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            fill_dtype_expr = (
                _torch_dtype_literal(str(fill_value_tensor.dtype))
                if fill_value_tensor is not None
                else f"{_tensor_expr(str(op.inputs[1]))}.dtype"
            )
            if (
                fill_shape_input_producer is not None
                and str(fill_shape_input_producer.op_type) == "SHAPE"
                and len(fill_shape_input_producer.inputs) >= 1
            ):
                fill_device_expr = f"{_tensor_expr(str(fill_shape_input_producer.inputs[0]))}.device"
            else:
                fill_device_expr = f"{_tensor_expr(str(op.inputs[0]))}.device"
            if fill_shape_values is not None:
                fill_shape_expr = repr([int(v) for v in list(fill_shape_values)])
                fill_shape_arg = f"[int(v) for v in {fill_shape_expr}]"
            elif (
                fill_shape_input_producer is not None
                and str(fill_shape_input_producer.op_type) == "SHAPE"
                and len(fill_shape_input_producer.inputs) >= 1
            ):
                runtime_imports.add("_tensor_shape_list")
                fill_shape_arg = f"_tensor_shape_list({_tensor_expr(str(fill_shape_input_producer.inputs[0]))})"
            else:
                reconstructed_fill_shape_expr = _reconstruct_shape_list_expr(fill_shape_input_name)
                if reconstructed_fill_shape_expr is not None:
                    fill_shape_arg = reconstructed_fill_shape_expr
                else:
                    runtime_imports.add("_shape_list")
                    fill_shape_arg = f"_shape_list({_tensor_expr(fill_shape_input_name)})"
            forward_lines.append(
                f"{output_vars[0]} = torch.full({fill_shape_arg}, {fill_value_expr}, dtype={fill_dtype_expr}, device={fill_device_expr})"
            )
            continue
        if op_type == "RANDOM_STANDARD_NORMAL":
            runtime_imports.add("_apply_random_standard_normal")
            random_output_tensor = model_ir.tensors.get(str(outputs[0]), None)
            random_output_shape_values = (
                [int(v) for v in list(random_output_tensor.shape)]
                if random_output_tensor is not None
                and all(int(v) > 0 for v in list(random_output_tensor.shape))
                else None
            )
            random_shape_values = (
                random_output_shape_values
                if random_output_shape_values is not None
                else _constant_int_list(model_ir.tensors.get(str(op.inputs[0]), None))
            )
            if random_shape_values is not None:
                runtime_imports.add("_module_device")
                random_shape_expr = repr([int(v) for v in list(random_shape_values)])
                random_device_expr = "_module_device(self)"
            else:
                runtime_imports.add("_shape_list")
                random_shape_expr = _tensor_expr(str(op.inputs[0]))
                random_device_expr = f"{_tensor_expr(str(op.inputs[0]))}.device"
            random_dtype_expr = (
                _torch_dtype_literal(str(random_output_tensor.dtype))
                if random_output_tensor is not None
                else "torch.float32"
            )
            random_seed_value = op.options.get("seed", None)
            random_seed_expr = (
                repr(int(random_seed_value))
                if random_seed_value is not None
                else "None"
            )
            forward_lines.append(
                f"{output_vars[0]} = _apply_random_standard_normal({random_shape_expr}, dtype={random_dtype_expr}, device={random_device_expr}, seed={random_seed_expr})"
            )
            continue
        if op_type == "SCATTER_ND":
            runtime_imports.add("_apply_scatter_nd")
            scatter_shape_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[2]), None))
            scatter_shape_expr = (
                repr([int(v) for v in list(scatter_shape_values)])
                if scatter_shape_values is not None
                else _tensor_expr(str(op.inputs[2]))
            )
            forward_lines.append(
                f"{output_vars[0]} = _apply_scatter_nd({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {scatter_shape_expr}, {output_target_shape})"
            )
            continue
        if op_type == "RANGE":
            if _range_only_feeds_identity_nms_postprocess_gathers(outputs[0]):
                continue
            start_tensor_expr = _tensor_expr(str(op.inputs[0]))
            limit_tensor_expr = _tensor_expr(str(op.inputs[1]))
            delta_tensor_expr = _tensor_expr(str(op.inputs[2]))
            start_expr = _scalar_literal_expr(str(op.inputs[0])) or start_tensor_expr
            limit_expr = _scalar_literal_expr(str(op.inputs[1])) or limit_tensor_expr
            delta_expr = _scalar_literal_expr(str(op.inputs[2])) or delta_tensor_expr
            start_value_expr = start_expr if _scalar_literal_expr(str(op.inputs[0])) is not None else f"{start_expr}.reshape(-1)[0].item()"
            limit_value_expr = limit_expr if _scalar_literal_expr(str(op.inputs[1])) is not None else f"{limit_expr}.reshape(-1)[0].item()"
            delta_value_expr = delta_expr if _scalar_literal_expr(str(op.inputs[2])) is not None else f"{delta_expr}.reshape(-1)[0].item()"
            range_device_expr = (
                f"{limit_tensor_expr}.device"
                if _scalar_literal_expr(str(op.inputs[1])) is None else
                f"{start_tensor_expr}.device"
                if _scalar_literal_expr(str(op.inputs[0])) is None else
                f"{delta_tensor_expr}.device"
                if _scalar_literal_expr(str(op.inputs[2])) is None else
                "_module_device(self)"
            )
            if (
                _scalar_literal_expr(str(op.inputs[0])) is not None
                and _scalar_literal_expr(str(op.inputs[1])) is not None
                and _scalar_literal_expr(str(op.inputs[2])) is not None
            ):
                runtime_imports.add("_module_device")
            range_dtype_expr = (
                f"{limit_tensor_expr}.dtype"
                if _scalar_literal_expr(str(op.inputs[1])) is None else
                f"{start_tensor_expr}.dtype"
                if _scalar_literal_expr(str(op.inputs[0])) is None else
                f"{delta_tensor_expr}.dtype"
                if _scalar_literal_expr(str(op.inputs[2])) is None else
                _torch_dtype_literal(str(model_ir.tensors[str(outputs[0])].dtype))
            )
            forward_lines.append(
                f"{output_vars[0]} = torch.arange(start={start_value_expr}, end={limit_value_expr}, step={delta_value_expr}, device={range_device_expr}, dtype={range_dtype_expr})"
            )
            continue
        if op_type == "CUMSUM":
            runtime_imports.add("_apply_cumsum")
            input_name = str(op.inputs[0])
            input_expr = _tensor_expr(input_name)
            axis_expr = (
                _axis_expr_from_input(str(op.inputs[1]), device_expr=input_expr)
                if len(op.inputs) >= 2
                else repr(int(op.options.get("axis", 0)))
            )
            expr = (
                f"_apply_cumsum({input_expr}, axis={axis_expr}, "
                f"exclusive={bool(op.options.get('exclusive', False))}, "
                f"reverse={bool(op.options.get('reverse', False))})"
            )
            inferred_shape = _tensor_shape_list(input_name)
            if _should_skip_align_for_shape_preserving_unary(input_name, outputs[0]):
                forward_lines.append(f"{output_vars[0]} = {expr}")
            else:
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=expr, inferred_shape=inferred_shape)}"
                )
            continue
        if op_type == "DEPTH_TO_SPACE":
            block_size = int(op.options.get("blockSize", 1))
            input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            input_shape = _tensor_shape_list(str(op.inputs[0]))
            output_shape = _tensor_shape_list(outputs[0])
            input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            output_layout = normalize_logical_layout(output_tensor.logical_layout) if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            use_channel_last = bool(
                is_channel_last_logical_layout(input_layout)
                or is_channel_last_logical_layout(output_layout)
            )
            inferred_channel_last = None
            if input_shape is not None and output_shape is not None:
                inferred_channel_last = _should_emit_channel_last_depth_to_space(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    block_size=block_size,
                )
            if inferred_channel_last is True:
                use_channel_last = True
            elif inferred_channel_last is False:
                use_channel_last = False
            input_expr = _tensor_expr(str(op.inputs[0]))
            if use_channel_last:
                forward_lines.append(f"_depth_to_space_x_{op_index} = {input_expr}")
                forward_lines.append(
                    f"_depth_to_space_n_{op_index}, _depth_to_space_h_{op_index}, _depth_to_space_w_{op_index}, _depth_to_space_c_{op_index} = _depth_to_space_x_{op_index}.shape"
                )
                forward_lines.append(
                    f"{output_vars[0]} = _depth_to_space_x_{op_index}.reshape(_depth_to_space_n_{op_index}, _depth_to_space_h_{op_index}, _depth_to_space_w_{op_index}, {block_size}, {block_size}, _depth_to_space_c_{op_index} // {block_size * block_size}).permute(0, 1, 3, 2, 4, 5).reshape(_depth_to_space_n_{op_index}, _depth_to_space_h_{op_index} * {block_size}, _depth_to_space_w_{op_index} * {block_size}, _depth_to_space_c_{op_index} // {block_size * block_size})"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'F.pixel_shuffle({input_expr}, {block_size})', inferred_shape=_tensor_shape_list(outputs[0]))}"
                )
            continue
        if op_type == "SPACE_TO_DEPTH":
            block_size = int(op.options.get("blockSize", 1))
            input_expr = _tensor_expr(str(op.inputs[0]))
            input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            input_shape = _tensor_shape_list(str(op.inputs[0]))
            output_shape = _tensor_shape_list(outputs[0])
            input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            output_layout = normalize_logical_layout(output_tensor.logical_layout) if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            use_channel_last = bool(
                is_channel_last_logical_layout(input_layout)
                or is_channel_last_logical_layout(output_layout)
            )
            inferred_channel_last = None
            if input_shape is not None and output_shape is not None:
                inferred_channel_last = _should_emit_channel_last_space_to_depth(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    block_size=block_size,
                )
            if inferred_channel_last is True:
                use_channel_last = True
            elif inferred_channel_last is False:
                use_channel_last = False
            forward_lines.append(f"_space_to_depth_x_{op_index} = {input_expr}")
            if use_channel_last:
                forward_lines.append(
                    f"_space_to_depth_n_{op_index}, _space_to_depth_h_{op_index}, _space_to_depth_w_{op_index}, _space_to_depth_c_{op_index} = _space_to_depth_x_{op_index}.shape"
                )
                forward_lines.append(
                    f"{output_vars[0]} = _space_to_depth_x_{op_index}.reshape(_space_to_depth_n_{op_index}, _space_to_depth_h_{op_index} // {block_size}, {block_size}, _space_to_depth_w_{op_index} // {block_size}, {block_size}, _space_to_depth_c_{op_index}).permute(0, 1, 3, 2, 4, 5).reshape(_space_to_depth_n_{op_index}, _space_to_depth_h_{op_index} // {block_size}, _space_to_depth_w_{op_index} // {block_size}, _space_to_depth_c_{op_index} * {block_size * block_size})"
                )
            else:
                forward_lines.append(
                    f"_space_to_depth_n_{op_index}, _space_to_depth_c_{op_index}, _space_to_depth_h_{op_index}, _space_to_depth_w_{op_index} = _space_to_depth_x_{op_index}.shape"
                )
                forward_lines.append(
                    f"{output_vars[0]} = _space_to_depth_x_{op_index}.reshape(_space_to_depth_n_{op_index}, _space_to_depth_c_{op_index}, _space_to_depth_h_{op_index} // {block_size}, {block_size}, _space_to_depth_w_{op_index} // {block_size}, {block_size}).permute(0, 1, 3, 5, 2, 4).reshape(_space_to_depth_n_{op_index}, _space_to_depth_c_{op_index} * {block_size * block_size}, _space_to_depth_h_{op_index} // {block_size}, _space_to_depth_w_{op_index} // {block_size})"
                )
            continue
        if op_type == "SOFTMAX":
            runtime_imports.add("_apply_softmax")
            axis = op.options.get("axis", None)
            if axis is None and len(op.inputs) > 0:
                input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                if input_tensor is not None:
                    input_layout = normalize_logical_layout(input_tensor.logical_layout)
                    if is_channel_first_logical_layout(input_layout) and len(list(input_tensor.shape)) >= 2:
                        axis = 1
            axis_expr = repr(int(axis)) if axis is not None else "None"
            beta = float(op.options.get("beta", 1.0))
            forward_lines.append(
                f"{output_vars[0]} = _apply_softmax({_tensor_expr(str(op.inputs[0]))}, axis={axis_expr}, beta={beta}, target_shape={output_target_shape})"
            )
            continue
        if op_type in {"ARG_MAX", "ARG_MIN"}:
            runtime_imports.add("_normalize_dim")
            input_expr = _tensor_expr(str(op.inputs[0]))
            if len(op.inputs) >= 2:
                axis_expr = _axis_expr_from_input(str(op.inputs[1]), device_expr=input_expr)
            else:
                axis_expr = repr(int(op.options.get("axis", 0)))
            input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            output_tensor = model_ir.tensors.get(outputs[0], None)
            keep_dims = bool(op.options.get("keepDims", False))
            if input_tensor is not None and output_tensor is not None:
                keep_dims = len(list(output_tensor.shape)) == len(list(input_tensor.shape))
            reduce_fn = "torch.argmax" if op_type == "ARG_MAX" else "torch.argmin"
            output_dtype = "INT64" if output_tensor is None else str(output_tensor.dtype)
            forward_lines.append(
                f"{output_vars[0]} = {reduce_fn}({input_expr}, dim=_normalize_dim({axis_expr}, {input_expr}.ndim), keepdim={keep_dims}).to(dtype={_torch_dtype_literal(output_dtype)})"
            )
            continue
        if op_type == "TOPK_V2":
            runtime_imports.add("_normalize_dim")
            input_expr = _tensor_expr(str(op.inputs[0]))
            topk_pre_perm, topk_index_post_perm = _topk_codegen_layout_bridge(
                input_name=str(op.inputs[0]),
                value_output_name=str(outputs[0]),
                index_output_name=str(outputs[1]) if len(outputs) > 1 else None,
            )
            if topk_pre_perm is not None:
                input_expr = f"{input_expr}.permute({', '.join(str(int(v)) for v in topk_pre_perm)}).contiguous()"
            input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            output_value_tensor = model_ir.tensors.get(outputs[0], None)
            output_index_tensor = model_ir.tensors.get(outputs[1], None) if len(outputs) > 1 else None
            k_literal = _int_scalar_literal_expr(str(op.inputs[1]))
            k_reconstructed_scalar_expr = None if k_literal is not None else _reconstruct_shape_scalar_expr(str(op.inputs[1]))
            k_expr = (
                repr(int(k_literal))
                if k_literal is not None else
                (
                    f"{k_reconstructed_scalar_expr}"
                    if k_reconstructed_scalar_expr is not None else
                f"int({_tensor_expr(str(op.inputs[1]))}.reshape(-1)[0].to(dtype=torch.int64).item())"
                )
            )
            axis_expr = repr(int(op.options.get("axis", -1)))
            largest = bool(op.options.get("largest", True))
            sorted_output = bool(op.options.get("sorted", True))
            output_value_shape_signature = (
                [int(v) for v in list(output_value_tensor.shape_signature)]
                if output_value_tensor is not None and output_value_tensor.shape_signature is not None
                else None
            )
            input_shape_signature = (
                [int(v) for v in list(input_tensor.shape_signature)]
                if input_tensor is not None and input_tensor.shape_signature is not None
                else None
            )
            can_emit_full_sort = (
                k_literal is None
                and
                output_value_shape_signature is not None
                and input_shape_signature is not None
                and output_value_shape_signature == input_shape_signature
            )
            if can_emit_full_sort:
                forward_lines.append(
                    f"{output_vars[0]}, {output_vars[1]} = torch.sort({input_expr}, dim=_normalize_dim({axis_expr}, {input_expr}.ndim), descending={largest})"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]}, {output_vars[1]} = torch.topk({input_expr}, k={k_expr}, dim=_normalize_dim({axis_expr}, {input_expr}.ndim), largest={largest}, sorted={sorted_output})"
                )
            runtime_shape_uncertain_tensors.update(outputs)
            if topk_index_post_perm is not None:
                forward_lines.append(
                    f"{output_vars[1]} = {output_vars[1]}.permute({', '.join(str(int(v)) for v in topk_index_post_perm)}).contiguous()"
                )
            if output_value_tensor is not None:
                forward_lines.append(
                    f"{output_vars[0]} = {output_vars[0]}.to(dtype={_torch_dtype_literal(str(output_value_tensor.dtype))})"
                )
            if output_index_tensor is not None:
                forward_lines.append(
                    f"{output_vars[1]} = {output_vars[1]}.to(dtype={_torch_dtype_literal(str(output_index_tensor.dtype))})"
                )
            continue
        if op_type == "AVERAGE_POOL_2D":
            options = dict(op.options)
            pool_input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            pool_output_tensor = model_ir.tensors.get(str(op.outputs[0]), None) if len(op.outputs) == 1 else None
            effective_layout = _infer_effective_rank4_runtime_layout(str(op.inputs[0]))
            pool_shape_implies_channel_last: Optional[bool] = None
            if pool_input_tensor is not None and pool_output_tensor is not None:
                input_shape = [int(v) for v in list(pool_input_tensor.shape)]
                output_shape = [int(v) for v in list(pool_output_tensor.shape)]
                if len(input_shape) == 4 and len(output_shape) == 4:
                    if input_shape[3] == output_shape[1] and input_shape[1] != output_shape[1]:
                        pool_shape_implies_channel_last = True
                    elif input_shape[1] == output_shape[1] and input_shape[3] != output_shape[1]:
                        pool_shape_implies_channel_last = False
            pool_alias_expr = channel_first_tensor_expr_aliases.get(str(op.inputs[0]), None)
            pool_distinct_channel_first_alias = (
                pool_alias_expr is not None
                and pool_alias_expr != _tensor_expr(str(op.inputs[0]))
            )
            pool_uses_channel_first_alias = (
                pool_distinct_channel_first_alias
            )
            pool_input_expr = (
                str(pool_alias_expr)
                if pool_uses_channel_first_alias else
                _tensor_expr(str(op.inputs[0]))
            )
            pool_as_channel_last_expr = (
                "False"
                if pool_uses_channel_first_alias else
                "True" if pool_shape_implies_channel_last is True else
                "False" if pool_shape_implies_channel_last is False else
                "True" if effective_layout == "NHWC" else
                "False" if effective_layout == "NCHW" else
                "None"
            )
            output_cf_shape = _rank4_channel_first_shape_for_tensor(outputs[0])
            can_emit_pool_cf_alias = (
                pool_as_channel_last_expr == "False"
                and output_cf_shape is not None
                and len(output_cf_shape) == 4
                and all(int(dim) > 0 for dim in output_cf_shape)
            )
            runtime_imports.add("_apply_pool2d")
            if can_emit_pool_cf_alias:
                raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
                forward_lines.append(
                    f"{raw_output_var} = _apply_pool2d({pool_input_expr}, "
                    f"filter_height={int(options.get('filterHeight', 1))}, "
                    f"filter_width={int(options.get('filterWidth', 1))}, "
                    f"stride_h={int(options.get('strideH', 1))}, "
                    f"stride_w={int(options.get('strideW', 1))}, "
                    f"padding={str(options.get('padding', 'VALID')).upper()!r}, "
                    f"target_shape={repr(output_cf_shape)}, "
                    f"is_max_pool=False, channel_last=False)"
                )
                if not _can_omit_materialized_channel_last_alias(outputs[0]):
                    runtime_imports.add("_align_tensor_to_target_shape")
                    forward_lines.append(
                        f"{output_vars[0]} = _align_tensor_to_target_shape({raw_output_var}.permute(0, 2, 3, 1).contiguous(), {output_target_shape})"
                    )
            else:
                channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                forward_lines.append(
                    f"{output_vars[0]} = _apply_pool2d({pool_input_expr}, "
                    f"filter_height={int(options.get('filterHeight', 1))}, "
                    f"filter_width={int(options.get('filterWidth', 1))}, "
                    f"stride_h={int(options.get('strideH', 1))}, "
                    f"stride_w={int(options.get('strideW', 1))}, "
                    f"padding={str(options.get('padding', 'VALID')).upper()!r}, "
                    f"target_shape={output_target_shape}, "
                    f"is_max_pool=False, channel_last={pool_as_channel_last_expr})"
                )
            forward_lines.extend(_activation_lines(output_vars[0], str(options.get("fusedActivationFunction", "NONE"))))
            continue
        if op_type == "MAX_POOL_2D":
            options = dict(op.options)
            pool_input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            pool_output_tensor = model_ir.tensors.get(str(op.outputs[0]), None) if len(op.outputs) == 1 else None
            effective_layout = _infer_effective_rank4_runtime_layout(str(op.inputs[0]))
            pool_shape_implies_channel_last: Optional[bool] = None
            if pool_input_tensor is not None and pool_output_tensor is not None:
                input_shape = [int(v) for v in list(pool_input_tensor.shape)]
                output_shape = [int(v) for v in list(pool_output_tensor.shape)]
                if len(input_shape) == 4 and len(output_shape) == 4:
                    if input_shape[3] == output_shape[1] and input_shape[1] != output_shape[1]:
                        pool_shape_implies_channel_last = True
                    elif input_shape[1] == output_shape[1] and input_shape[3] != output_shape[1]:
                        pool_shape_implies_channel_last = False
            pool_alias_expr = channel_first_tensor_expr_aliases.get(str(op.inputs[0]), None)
            pool_distinct_channel_first_alias = (
                pool_alias_expr is not None
                and pool_alias_expr != _tensor_expr(str(op.inputs[0]))
            )
            pool_uses_channel_first_alias = (
                pool_distinct_channel_first_alias
            )
            pool_input_expr = (
                str(pool_alias_expr)
                if pool_uses_channel_first_alias else
                _tensor_expr(str(op.inputs[0]))
            )
            pool_as_channel_last_expr = (
                "False"
                if pool_uses_channel_first_alias else
                "True" if pool_shape_implies_channel_last is True else
                "False" if pool_shape_implies_channel_last is False else
                "True" if effective_layout == "NHWC" else
                "False" if effective_layout == "NCHW" else
                "None"
            )
            output_cf_shape = _rank4_channel_first_shape_for_tensor(outputs[0])
            can_emit_pool_cf_alias = (
                pool_as_channel_last_expr == "False"
                and output_cf_shape is not None
                and len(output_cf_shape) == 4
                and all(int(dim) > 0 for dim in output_cf_shape)
            )
            runtime_imports.add("_apply_pool2d")
            if can_emit_pool_cf_alias:
                raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
                forward_lines.append(
                    f"{raw_output_var} = _apply_pool2d({pool_input_expr}, "
                    f"filter_height={int(options.get('filterHeight', 1))}, "
                    f"filter_width={int(options.get('filterWidth', 1))}, "
                    f"stride_h={int(options.get('strideH', 1))}, "
                    f"stride_w={int(options.get('strideW', 1))}, "
                    f"padding={str(options.get('padding', 'VALID')).upper()!r}, "
                    f"target_shape={repr(output_cf_shape)}, "
                    f"is_max_pool=True, channel_last=False)"
                )
                if not _can_omit_materialized_channel_last_alias(outputs[0]):
                    runtime_imports.add("_align_tensor_to_target_shape")
                    forward_lines.append(
                        f"{output_vars[0]} = _align_tensor_to_target_shape({raw_output_var}.permute(0, 2, 3, 1).contiguous(), {output_target_shape})"
                    )
            else:
                channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                forward_lines.append(
                    f"{output_vars[0]} = _apply_pool2d({pool_input_expr}, "
                    f"filter_height={int(options.get('filterHeight', 1))}, "
                    f"filter_width={int(options.get('filterWidth', 1))}, "
                    f"stride_h={int(options.get('strideH', 1))}, "
                    f"stride_w={int(options.get('strideW', 1))}, "
                    f"padding={str(options.get('padding', 'VALID')).upper()!r}, "
                    f"target_shape={output_target_shape}, "
                    f"is_max_pool=True, channel_last={pool_as_channel_last_expr})"
                )
            forward_lines.extend(_activation_lines(output_vars[0], str(options.get("fusedActivationFunction", "NONE"))))
            continue
        if op_type == "RESIZE_NEAREST_NEIGHBOR":
            size_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            size_literal = _python_literal_for_constant_tensor(size_tensor) if size_tensor is not None else None
            size_values = _constant_int_list(size_tensor) if size_tensor is not None else None
            resize_target_shape = _resize_target_shape_literal(outputs[0], str(op.inputs[0]))
            resize_input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            resize_output_tensor = model_ir.tensors.get(str(outputs[0]), None) if len(outputs) == 1 else None
            resize_effective_layout = _infer_effective_rank4_runtime_layout(str(op.inputs[0]))
            resize_shape_implies_channel_last: Optional[bool] = None
            if resize_input_tensor is not None and resize_output_tensor is not None:
                resize_input_shape = [int(v) for v in list(resize_input_tensor.shape)]
                resize_output_shape = [int(v) for v in list(resize_output_tensor.shape)]
                if len(resize_input_shape) == 4 and len(resize_output_shape) == 4:
                    if resize_input_shape[3] == resize_output_shape[3] and resize_input_shape[1] != resize_output_shape[3]:
                        resize_shape_implies_channel_last = True
                    elif resize_input_shape[1] == resize_output_shape[1] and resize_input_shape[3] != resize_output_shape[1]:
                        resize_shape_implies_channel_last = False
            resize_alias_expr = channel_first_tensor_expr_aliases.get(str(op.inputs[0]), None)
            resize_distinct_channel_first_alias = (
                resize_alias_expr is not None
                and resize_alias_expr != _tensor_expr(str(op.inputs[0]))
            )
            resize_uses_channel_first_alias = (
                resize_distinct_channel_first_alias
            )
            resize_input_expr = (
                str(resize_alias_expr)
                if resize_uses_channel_first_alias else
                _tensor_expr(str(op.inputs[0]))
            )
            output_cf_shape = _rank4_channel_first_shape_for_tensor(outputs[0])
            can_emit_resize_cf_alias = (
                resize_uses_channel_first_alias
                and output_cf_shape is not None
                and len(output_cf_shape) == 4
                and all(int(dim) > 0 for dim in output_cf_shape)
            )
            resize_channel_last_expr = (
                "False"
                if resize_uses_channel_first_alias else
                "True" if resize_shape_implies_channel_last is True else
                "False" if resize_shape_implies_channel_last is False else
                "True" if resize_effective_layout == "NHWC" else
                "False" if resize_effective_layout == "NCHW" else
                "None"
            )
            size_expr = (
                f"{size_literal}"
                if size_literal is not None else
                _tensor_expr(str(op.inputs[1]))
            )
            emitted_direct_resize = False
            input_cf_shape = _rank4_channel_first_shape_for_tensor(str(op.inputs[0]))
            output_shape = _tensor_shape_list(outputs[0])
            if resize_channel_last_expr == "False" and input_cf_shape is not None and size_values is not None and len(size_values) == 2:
                raw_resize_shape = [int(input_cf_shape[0]), int(input_cf_shape[1]), int(size_values[0]), int(size_values[1])]
                resize_expr = f"F.interpolate({resize_input_expr}, size={repr([int(size_values[0]), int(size_values[1])])}, mode='nearest')"
                if output_shape == raw_resize_shape:
                    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                    forward_lines.append(f"{output_vars[0]} = {resize_expr}")
                    emitted_direct_resize = True
                elif output_shape == [int(raw_resize_shape[0]), int(raw_resize_shape[2]), int(raw_resize_shape[3]), int(raw_resize_shape[1])]:
                    raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                    channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
                    forward_lines.append(f"{raw_output_var} = {resize_expr}")
                    if not _can_omit_materialized_channel_last_alias(outputs[0]):
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_vars[0]} = _align_tensor_to_target_shape({raw_output_var}.permute(0, 2, 3, 1).contiguous(), {resize_target_shape})"
                        )
                    emitted_direct_resize = True
            if not emitted_direct_resize:
                runtime_imports.add("_apply_resize")
                if can_emit_resize_cf_alias:
                    raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                    channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
                    forward_lines.append(
                        f"{raw_output_var} = _apply_resize({resize_input_expr}, {size_expr}, method='nearest', target_shape={repr(output_cf_shape)}, channel_last=False)"
                    )
                    if not _can_omit_materialized_channel_last_alias(outputs[0]):
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_vars[0]} = _align_tensor_to_target_shape({raw_output_var}.permute(0, 2, 3, 1).contiguous(), {resize_target_shape})"
                        )
                else:
                    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                    forward_lines.append(
                        f"{output_vars[0]} = _apply_resize({resize_input_expr}, {size_expr}, method='nearest', target_shape={resize_target_shape}, channel_last={resize_channel_last_expr})"
                    )
            continue
        if op_type == "RESIZE_BILINEAR":
            size_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            size_literal = _python_literal_for_constant_tensor(size_tensor) if size_tensor is not None else None
            size_values = _constant_int_list(size_tensor) if size_tensor is not None else None
            align_corners = bool(op.options.get("alignCorners", False))
            half_pixel_centers = bool(op.options.get("halfPixelCenters", False))
            resize_target_shape = _resize_target_shape_literal(outputs[0], str(op.inputs[0]))
            resize_input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            resize_output_tensor = model_ir.tensors.get(str(outputs[0]), None) if len(outputs) == 1 else None
            resize_effective_layout = _infer_effective_rank4_runtime_layout(str(op.inputs[0]))
            resize_shape_implies_channel_last: Optional[bool] = None
            if resize_input_tensor is not None and resize_output_tensor is not None:
                resize_input_shape = [int(v) for v in list(resize_input_tensor.shape)]
                resize_output_shape = [int(v) for v in list(resize_output_tensor.shape)]
                if len(resize_input_shape) == 4 and len(resize_output_shape) == 4:
                    if resize_input_shape[3] == resize_output_shape[3] and resize_input_shape[1] != resize_output_shape[3]:
                        resize_shape_implies_channel_last = True
                    elif resize_input_shape[1] == resize_output_shape[1] and resize_input_shape[3] != resize_output_shape[1]:
                        resize_shape_implies_channel_last = False
            resize_alias_expr = channel_first_tensor_expr_aliases.get(str(op.inputs[0]), None)
            resize_distinct_channel_first_alias = (
                resize_alias_expr is not None
                and resize_alias_expr != _tensor_expr(str(op.inputs[0]))
            )
            resize_uses_channel_first_alias = (
                resize_distinct_channel_first_alias
            )
            resize_input_expr = (
                str(resize_alias_expr)
                if resize_uses_channel_first_alias else
                _tensor_expr(str(op.inputs[0]))
            )
            output_cf_shape = _rank4_channel_first_shape_for_tensor(outputs[0])
            can_emit_resize_cf_alias = (
                resize_uses_channel_first_alias
                and output_cf_shape is not None
                and len(output_cf_shape) == 4
                and all(int(dim) > 0 for dim in output_cf_shape)
            )
            resize_channel_last_expr = (
                "False"
                if resize_uses_channel_first_alias else
                "True" if resize_shape_implies_channel_last is True else
                "False" if resize_shape_implies_channel_last is False else
                "True" if resize_effective_layout == "NHWC" else
                "False" if resize_effective_layout == "NCHW" else
                "None"
            )
            size_expr = (
                f"{size_literal}"
                if size_literal is not None else
                _tensor_expr(str(op.inputs[1]))
            )
            emitted_direct_resize = False
            input_cf_shape = _rank4_channel_first_shape_for_tensor(str(op.inputs[0]))
            output_shape = _tensor_shape_list(outputs[0])
            if (
                resize_channel_last_expr == "False"
                and not half_pixel_centers
                and input_cf_shape is not None
                and size_values is not None
                and len(size_values) == 2
            ):
                raw_resize_shape = [int(input_cf_shape[0]), int(input_cf_shape[1]), int(size_values[0]), int(size_values[1])]
                resize_expr = (
                    f"F.interpolate({resize_input_expr}, size={repr([int(size_values[0]), int(size_values[1])])}, "
                    f"mode='bilinear', align_corners={align_corners})"
                )
                if output_shape == raw_resize_shape:
                    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                    forward_lines.append(f"{output_vars[0]} = {resize_expr}")
                    emitted_direct_resize = True
                elif output_shape == [int(raw_resize_shape[0]), int(raw_resize_shape[2]), int(raw_resize_shape[3]), int(raw_resize_shape[1])]:
                    raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                    channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
                    forward_lines.append(f"{raw_output_var} = {resize_expr}")
                    if not _can_omit_materialized_channel_last_alias(outputs[0]):
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_vars[0]} = _align_tensor_to_target_shape({raw_output_var}.permute(0, 2, 3, 1).contiguous(), {resize_target_shape})"
                        )
                    emitted_direct_resize = True
            if not emitted_direct_resize:
                runtime_imports.add("_apply_resize")
                if can_emit_resize_cf_alias:
                    raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                    channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
                    forward_lines.append(
                        f"{raw_output_var} = _apply_resize({resize_input_expr}, {size_expr}, method='bilinear', target_shape={repr(output_cf_shape)}, align_corners={align_corners}, half_pixel_centers={half_pixel_centers}, channel_last=False)"
                    )
                    if not _can_omit_materialized_channel_last_alias(outputs[0]):
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_vars[0]} = _align_tensor_to_target_shape({raw_output_var}.permute(0, 2, 3, 1).contiguous(), {resize_target_shape})"
                        )
                else:
                    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
                    forward_lines.append(
                        f"{output_vars[0]} = _apply_resize({resize_input_expr}, {size_expr}, method='bilinear', target_shape={resize_target_shape}, align_corners={align_corners}, half_pixel_centers={half_pixel_centers}, channel_last={resize_channel_last_expr})"
                    )
            continue
        if op_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            runtime_imports.update({"_normalize_axes"})
            reducer_map = {
                "SUM": "_reduce_sum",
                "MEAN": "_reduce_mean",
                "REDUCE_MAX": "_reduce_max",
                "REDUCE_MIN": "_reduce_min",
                "REDUCE_PROD": "_reduce_prod",
                "REDUCE_ANY": "_reduce_any",
            }
            runtime_imports.add(reducer_map[op_type])
            axis_values = None
            if len(op.inputs) >= 2:
                axis_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
            axis_expr = (
                f"_normalize_axes({repr(axis_values)}, {_tensor_expr(str(op.inputs[0]))}.ndim)"
                if axis_values is not None
                else f"_normalize_axes({_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[0]))}.ndim)"
                if len(op.inputs) >= 2
                else "None"
            )
            keepdims = bool(op.options.get("keepDims", True))
            reduction_output_name = str(outputs[0])
            reduction_output_tensor = model_ir.tensors.get(reduction_output_name, None)
            reduction_output_rank = len(list(reduction_output_tensor.shape)) if reduction_output_tensor is not None else 0
            reduction_output_layout = normalize_logical_layout(
                reduction_output_tensor.logical_layout if reduction_output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            )
            reduction_raw_output_var = output_vars[0]
            reduction_input_name = str(op.inputs[0])
            reduction_input_alias = channel_first_tensor_expr_aliases.get(reduction_input_name, None)
            reduction_input_expr = _tensor_expr(reduction_input_name)
            reduction_input_var = tensor_var_names.get(reduction_input_name, reduction_input_expr)
            reduction_uses_channel_first_alias = (
                reduction_input_alias is not None
                or reduction_input_expr != reduction_input_var
            )
            reduction_shape_implies_channel_last = (
                reduction_output_tensor is not None
                and reduction_output_rank == 4
                and int(reduction_output_tensor.shape[1]) == 1
                and int(reduction_output_tensor.shape[2]) == 1
                and int(reduction_output_tensor.shape[3]) > 1
            )
            reduction_input_tensor = model_ir.tensors.get(reduction_input_name, None)
            reduction_input_is_ambiguous_channel_last_gap = (
                reduction_input_tensor is not None
                and len(list(reduction_input_tensor.shape)) == 4
                and is_channel_last_logical_layout(
                    normalize_logical_layout(reduction_input_tensor.logical_layout)
                )
                and axis_values is not None
                and [int(v) for v in list(axis_values)] == [1, 2]
                and len(
                    set(
                        int(dim)
                        for dim in list(reduction_input_tensor.shape)[1:]
                        if int(dim) > 0
                    )
                )
                != len(
                    [
                        int(dim)
                        for dim in list(reduction_input_tensor.shape)[1:]
                        if int(dim) > 0
                    ]
                )
            )
            if (
                op_type == "MEAN"
                and keepdims
                and reduction_input_is_ambiguous_channel_last_gap
                and reduction_input_alias is not None
            ):
                inferred_reduction_shape = _infer_reduction_shape(
                    _tensor_shape_list(str(op.inputs[0])),
                    axis_values,
                    keepdims=keepdims,
                )
                reduced_expr = _emit_maybe_aligned_expr(
                    output_name=outputs[0],
                    expr=(
                        f"torch.mean({reduction_input_alias}.permute(0, 2, 3, 1).contiguous(), "
                        f"dim=[1, 2], keepdim=True)"
                    ),
                    inferred_shape=inferred_reduction_shape,
                )
                channel_first_tensor_expr_aliases.pop(reduction_output_name, None)
                forward_lines.append(f"{output_vars[0]} = {reduced_expr}")
                continue
            reduction_is_channel_last_gap = (
                op_type == "MEAN"
                and keepdims
                and axis_values is not None
                and [int(v) for v in list(axis_values)] == [1, 2]
                and reduction_shape_implies_channel_last
                and not reduction_input_is_ambiguous_channel_last_gap
            )
            if reduction_is_channel_last_gap:
                channel_first_tensor_expr_aliases[reduction_output_name] = output_vars[0]
            if reduction_is_channel_last_gap:
                reduction_input_expr = _tensor_expr(str(op.inputs[0]))
                if reduction_input_expr != tensor_var_names.get(str(op.inputs[0]), reduction_input_expr):
                    runtime_imports.add("_align_tensor_to_target_shape")
                    forward_lines.append(
                        f"{output_vars[0]} = _align_tensor_to_target_shape("
                        f"torch.mean({reduction_input_expr}, dim=[2, 3], keepdim=True).permute(0, 2, 3, 1).contiguous(), "
                        f"{_target_shape_literal(reduction_output_name)})"
                    )
                    channel_first_tensor_expr_aliases.pop(reduction_output_name, None)
                    continue
            if (
                (reduction_uses_channel_first_alias or reduction_is_channel_last_gap)
                and reduction_output_tensor is not None
                and (
                    is_channel_last_logical_layout(reduction_output_layout)
                    or reduction_shape_implies_channel_last
                )
                and reduction_output_rank in {3, 4, 5}
            ):
                reduction_raw_output_var = _derived_local_var_name(f"{output_vars[0]}_cf")
                channel_first_tensor_expr_aliases[reduction_output_name] = reduction_raw_output_var
            else:
                channel_first_tensor_expr_aliases.pop(reduction_output_name, None)
            channel_first_reduction_plan = _channel_first_reduction_plan(op, str(op.inputs[0]))
            if channel_first_reduction_plan is not None:
                input_expr, channel_first_axes = channel_first_reduction_plan
                if op_type == "MEAN":
                    direct_expr = _direct_mean_reduction_expr(
                        input_expr=input_expr,
                        axes=channel_first_axes,
                        input_rank=len(list(model_ir.tensors[str(op.inputs[0])].shape)),
                        keepdims=keepdims,
                    )
                    if direct_expr is not None:
                        reduced_expr = _emit_maybe_aligned_expr(
                            output_name=outputs[0],
                            expr=direct_expr,
                            inferred_shape=_infer_reduction_shape(_tensor_shape_list(str(op.inputs[0])), axis_values, keepdims=keepdims),
                        )
                        forward_lines.append(f"{reduction_raw_output_var} = {reduced_expr}")
                        if reduction_raw_output_var != output_vars[0] and not _can_omit_materialized_channel_last_alias(reduction_output_name):
                            runtime_imports.add("_align_tensor_to_target_shape")
                            forward_lines.append(
                                f"{output_vars[0]} = _align_tensor_to_target_shape("
                                f"{reduction_raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                                f"{_target_shape_literal(reduction_output_name)})"
                            )
                        continue
                axis_expr = f"_normalize_axes({repr(channel_first_axes)}, {input_expr}.ndim)"
                reduced_expr = _emit_maybe_aligned_expr(
                    output_name=outputs[0],
                    expr=f"{reducer_map[op_type]}({input_expr}, {axis_expr}, {keepdims})",
                    inferred_shape=_infer_reduction_shape(_tensor_shape_list(str(op.inputs[0])), axis_values, keepdims=keepdims),
                )
                forward_lines.append(f"{reduction_raw_output_var} = {reduced_expr}")
                if reduction_raw_output_var != output_vars[0] and not _can_omit_materialized_channel_last_alias(reduction_output_name):
                    runtime_imports.add("_align_tensor_to_target_shape")
                    forward_lines.append(
                        f"{output_vars[0]} = _align_tensor_to_target_shape("
                        f"{reduction_raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                        f"{_target_shape_literal(reduction_output_name)})"
                    )
                continue
            if op_type == "MEAN":
                direct_expr = _direct_mean_reduction_expr(
                    input_expr=_tensor_expr(str(op.inputs[0])),
                    axes=axis_values,
                    input_rank=len(list(model_ir.tensors[str(op.inputs[0])].shape)),
                    keepdims=keepdims,
                )
                if direct_expr is not None:
                    reduced_expr = _emit_maybe_aligned_expr(
                        output_name=outputs[0],
                        expr=direct_expr,
                        inferred_shape=_infer_reduction_shape(_tensor_shape_list(str(op.inputs[0])), axis_values, keepdims=keepdims),
                    )
                    forward_lines.append(f"{reduction_raw_output_var} = {reduced_expr}")
                    if reduction_raw_output_var != output_vars[0] and not _can_omit_materialized_channel_last_alias(reduction_output_name):
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_vars[0]} = _align_tensor_to_target_shape("
                            f"{reduction_raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                            f"{_target_shape_literal(reduction_output_name)})"
                        )
                    continue
            reduced_expr = _emit_maybe_aligned_expr(
                output_name=outputs[0],
                expr=f"{reducer_map[op_type]}({_tensor_expr(str(op.inputs[0]))}, {axis_expr}, {keepdims})",
                inferred_shape=_infer_reduction_shape(_tensor_shape_list(str(op.inputs[0])), axis_values, keepdims=keepdims),
            )
            forward_lines.append(f"{reduction_raw_output_var} = {reduced_expr}")
            if reduction_raw_output_var != output_vars[0] and not _can_omit_materialized_channel_last_alias(reduction_output_name):
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_vars[0]} = _align_tensor_to_target_shape("
                    f"{reduction_raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                    f"{_target_shape_literal(reduction_output_name)})"
                )
            continue
        if op_type == "PAD":
            runtime_imports.add("_align_tensor_to_target_shape")
            pad_input_expr = f"_align_tensor_to_target_shape({_tensor_expr(str(op.inputs[0]))}, {_target_shape_literal(str(op.inputs[0]))})"
            pad_literal_expr = _pad_literal_expr(str(op.inputs[1]))
            if pad_literal_expr is not None:
                forward_lines.append(
                    f"{output_vars[0]} = F.pad({pad_input_expr}, {pad_literal_expr}, mode='constant', value=0.0)"
                )
            else:
                runtime_imports.add("_apply_pad_nd")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_pad_nd({pad_input_expr}, {_tensor_expr(str(op.inputs[1]))}, mode='constant', value=0.0)"
                )
            continue
        if op_type == "PADV2":
            runtime_imports.add("_align_tensor_to_target_shape")
            pad_input_expr = f"_align_tensor_to_target_shape({_tensor_expr(str(op.inputs[0]))}, {_target_shape_literal(str(op.inputs[0]))})"
            value_expr = _scalar_literal_expr(str(op.inputs[2])) or f"float({_tensor_expr(str(op.inputs[2]))}.reshape(-1)[0].item())"
            pad_literal_expr = _pad_literal_expr(str(op.inputs[1]))
            if pad_literal_expr is not None:
                forward_lines.append(
                    f"{output_vars[0]} = F.pad({pad_input_expr}, {pad_literal_expr}, mode='constant', value={value_expr})"
                )
            else:
                runtime_imports.add("_apply_pad_nd")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_pad_nd({pad_input_expr}, {_tensor_expr(str(op.inputs[1]))}, mode='constant', value={value_expr})"
                )
            continue
        if op_type == "MIRROR_PAD":
            static_mirror_pad_expr = _static_mirror_pad_expr(
                input_tensor_name=str(op.inputs[0]),
                pads_tensor_name=str(op.inputs[1]),
            )
            if static_mirror_pad_expr is not None:
                forward_lines.append(f"{output_vars[0]} = {static_mirror_pad_expr}")
            else:
                runtime_imports.add("_apply_pad_nd")
                runtime_imports.add("_align_tensor_to_target_shape")
                pad_input_expr = f"_align_tensor_to_target_shape({_tensor_expr(str(op.inputs[0]))}, {_target_shape_literal(str(op.inputs[0]))})"
                forward_lines.append(
                    f"{output_vars[0]} = _apply_pad_nd({pad_input_expr}, {_tensor_expr(str(op.inputs[1]))}, mode='reflect')"
                )
            continue
        if op_type in {"WHERE", "SELECT", "SELECT_V2"}:
            if len(op.inputs) == 1:
                forward_lines.append(
                    f"{output_vars[0]} = torch.nonzero({_tensor_expr(str(op.inputs[0]))})"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = torch.where({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))})"
                )
            continue
        if op_type == "TILE":
            runtime_imports.add("_apply_tile")
            forward_lines.append(
                f"{output_vars[0]} = _apply_tile({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))})"
            )
            continue
        if op_type == "BATCH_MATMUL":
            x_expr = _tensor_expr(str(op.inputs[0]))
            y_expr = _tensor_expr(str(op.inputs[1]))
            adj_x = bool(op.options.get("adjX", False))
            adj_y = bool(op.options.get("adjY", False))
            effective_adj_x = bool(adj_x)
            effective_adj_y = bool(adj_y)
            x_shape = _tensor_shape_list(str(op.inputs[0]))
            y_shape = _tensor_shape_list(str(op.inputs[1]))
            output_shape = _tensor_shape_list(outputs[0])
            lhs_producer_idx = producer_index.get(str(op.inputs[0]), None)
            if lhs_producer_idx is not None:
                lhs_producer = model_ir.operators[int(lhs_producer_idx)]
                if str(lhs_producer.op_type) == "RESHAPE" and len(lhs_producer.inputs) >= 1 and len(lhs_producer.outputs) == 1:
                    lhs_feature_last_target = _reshape_prefers_feature_last_for_adjx_batch_matmul(
                        str(lhs_producer.inputs[0]),
                        str(lhs_producer.outputs[0]),
                    )
                    if lhs_feature_last_target is not None:
                        x_shape = [int(v) for v in list(lhs_feature_last_target[1])]
            inferred_shape = _infer_batch_matmul_shape(
                x_shape,
                y_shape,
                adj_x=effective_adj_x,
                adj_y=effective_adj_y,
            )
            if output_shape is not None and not _shape_lists_equal(inferred_shape, output_shape):
                best_flag_choice: Optional[Tuple[int, bool, bool, List[int]]] = None
                for candidate_adj_x, candidate_adj_y in [
                    (effective_adj_x, effective_adj_y),
                    (effective_adj_x, not effective_adj_y),
                    (not effective_adj_x, effective_adj_y),
                    (not effective_adj_x, not effective_adj_y),
                ]:
                    candidate_shape = _infer_batch_matmul_shape(
                        x_shape,
                        y_shape,
                        adj_x=candidate_adj_x,
                        adj_y=candidate_adj_y,
                    )
                    if not _shape_lists_equal(candidate_shape, output_shape):
                        continue
                    score = int(candidate_adj_x != adj_x) + int(candidate_adj_y != adj_y)
                    choice = (int(score), bool(candidate_adj_x), bool(candidate_adj_y), [int(v) for v in list(candidate_shape or [])])
                    if best_flag_choice is None or choice < best_flag_choice:
                        best_flag_choice = choice
                if best_flag_choice is not None:
                    _, effective_adj_x, effective_adj_y, inferred_shape = best_flag_choice
            emitted_x_expr = x_expr
            emitted_y_expr = y_expr
            runtime_adj_x = bool(effective_adj_x)
            runtime_adj_y = bool(effective_adj_y)
            if runtime_adj_x:
                transposed_x_expr = _transposed_constant_expr_for_tensor_name(str(op.inputs[0]))
                if transposed_x_expr is not None:
                    emitted_x_expr = str(transposed_x_expr)
                    runtime_adj_x = False
            if runtime_adj_y:
                transposed_y_expr = _transposed_constant_expr_for_tensor_name(str(op.inputs[1]))
                if transposed_y_expr is not None:
                    emitted_y_expr = str(transposed_y_expr)
                    runtime_adj_y = False
            forward_lines.append(f"_tmp_x_{op_index} = {emitted_x_expr}")
            forward_lines.append(f"_tmp_y_{op_index} = {emitted_y_expr}")
            if runtime_adj_x:
                forward_lines.append(f"_tmp_x_{op_index} = _tmp_x_{op_index}.transpose(-1, -2)")
            if runtime_adj_y:
                forward_lines.append(f"_tmp_y_{op_index} = _tmp_y_{op_index}.transpose(-1, -2)")
            if (
                not effective_adj_x
                and not effective_adj_y
                and x_shape is not None
                and y_shape is not None
                and len(x_shape) >= 3
                and len(y_shape) == 2
            ):
                transposed_x_shape = list(x_shape[:-2]) + [int(x_shape[-1]), int(x_shape[-2])]
                inferred_shape_with_x_transpose = _infer_batch_matmul_shape(
                    transposed_x_shape,
                    y_shape,
                    adj_x=False,
                    adj_y=False,
                )
                input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
                if (
                    inferred_shape is None
                    and inferred_shape_with_x_transpose is not None
                    and is_channel_first_logical_layout(input_layout)
                ):
                    forward_lines.append(f"_tmp_x_{op_index} = _tmp_x_{op_index}.transpose(-1, -2)")
                    inferred_shape = inferred_shape_with_x_transpose
            runtime_imports.add("_align_tensor_to_target_shape")
            if output_target_shape != "None":
                forward_lines.append(
                    f"{output_vars[0]} = _align_tensor_to_target_shape(torch.matmul(_tmp_x_{op_index}, _tmp_y_{op_index}), {output_target_shape})"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'torch.matmul(_tmp_x_{op_index}, _tmp_y_{op_index})', inferred_shape=inferred_shape)}"
                )
            continue
        if op_type == "NON_MAX_SUPPRESSION_V4":
            runtime_imports.add("_apply_non_max_suppression_v4")
            nms_method_name = f"_run_nms_{len(nms_method_specs)}"
            max_output_literal = _int_scalar_literal_expr(str(op.inputs[2]))
            max_output_shape_expr = None if max_output_literal is not None else _reconstruct_shape_scalar_expr(str(op.inputs[2]))
            if max_output_shape_expr is not None:
                max_output_shape_expr = max_output_shape_expr.replace(_tensor_expr(str(op.inputs[0])), "boxes")
                max_output_shape_expr = max_output_shape_expr.replace(_tensor_expr(str(op.inputs[1])), "scores")
            iou_threshold_literal = _scalar_literal_expr(str(op.inputs[3]))
            score_threshold_literal = _scalar_literal_expr(str(op.inputs[4]))
            nms_method_specs.append(
                {
                    "name": nms_method_name,
                    "max_output_expr": max_output_literal or max_output_shape_expr or "max_output_size",
                    "iou_threshold_expr": iou_threshold_literal or "iou_threshold",
                    "score_threshold_expr": score_threshold_literal or "score_threshold",
                    "max_output_arg_expr": None if (max_output_literal is not None or max_output_shape_expr is not None) else _tensor_expr(str(op.inputs[2])),
                    "iou_threshold_arg_expr": None if iou_threshold_literal is not None else _tensor_expr(str(op.inputs[3])),
                    "score_threshold_arg_expr": None if score_threshold_literal is not None else _tensor_expr(str(op.inputs[4])),
                }
            )
            call_args = [
                _tensor_expr(str(op.inputs[0])),
                _tensor_expr(str(op.inputs[1])),
            ]
            if max_output_literal is None and max_output_shape_expr is None:
                call_args.append(_tensor_expr(str(op.inputs[2])))
            if iou_threshold_literal is None:
                call_args.append(_tensor_expr(str(op.inputs[3])))
            if score_threshold_literal is None:
                call_args.append(_tensor_expr(str(op.inputs[4])))
            forward_lines.append(
                f"{', '.join(output_vars)} = self.{nms_method_name}({', '.join(call_args)})"
            )
            runtime_shape_uncertain_tensors.update(outputs)
            if len(outputs) >= 2 and len(output_vars) >= 2 and len(consumer_index.get(outputs[0], [])) > 0:
                runtime_imports.add("_crop_nms_selected_indices")
                cropped_indices_var = f"_nms_selected_indices_valid_{op_index}"
                forward_lines.append(
                    f"{cropped_indices_var} = _crop_nms_selected_indices({output_vars[0]}, {output_vars[1]})"
                )
                tensor_expr_aliases[outputs[0]] = cropped_indices_var
            continue
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen hit an unimplemented op emitter. "
            f"op_type={op_type}"
        )

    has_conv_blocks = len(fused_module_specs) > 0
    conv_block_helper_source = (
        "class _Conv2dBlock(torch.nn.Module):\n"
        "    def __init__(\n"
        "        self,\n"
        "        *,\n"
        "        in_channels: int,\n"
        "        out_channels: int,\n"
        "        kernel_size: tuple[int, int],\n"
        "        stride: tuple[int, int],\n"
        "        padding: tuple[int, int],\n"
        "        dilation: tuple[int, int],\n"
        "        groups: int,\n"
        "        bias: bool,\n"
        "        pad: Optional[list[int]] = None,\n"
        "        activation: str = 'none',\n"
        "        negative_slope: float = 0.2,\n"
        "        pad_mode: str = 'constant',\n"
        "        pad_value: float = 0.0,\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "        self.conv = torch.nn.Conv2d(\n"
        "            in_channels=in_channels,\n"
        "            out_channels=out_channels,\n"
        "            kernel_size=kernel_size,\n"
        "            stride=stride,\n"
        "            padding=padding,\n"
        "            dilation=dilation,\n"
        "            groups=groups,\n"
        "            bias=bias,\n"
        "        )\n"
        "        self.pad = pad\n"
        "        self.activation = str(activation)\n"
        "        self.negative_slope = float(negative_slope)\n"
        "        self.pad_mode = str(pad_mode)\n"
        "        self.pad_value = float(pad_value)\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        input_was_channel_last = False\n"
        "        if x.ndim == 4 and int(x.shape[1]) != int(self.conv.in_channels) and int(x.shape[-1]) == int(self.conv.in_channels):\n"
        "            x = x.permute(0, 3, 1, 2).contiguous()\n"
        "            input_was_channel_last = True\n"
        "        if self.pad is not None:\n"
        "            x = F.pad(x, self.pad, mode=self.pad_mode, value=self.pad_value)\n"
        "        x = self.conv(x)\n"
        "        if input_was_channel_last:\n"
        "            x = x.permute(0, 2, 3, 1).contiguous()\n"
        "        if self.activation == 'leaky_relu':\n"
        "            return F.leaky_relu(x, negative_slope=self.negative_slope)\n"
        "        if self.activation == 'relu':\n"
        "            return torch.relu(x)\n"
        "        if self.activation == 'relu6':\n"
        "            return torch.clamp(x, min=0.0, max=6.0)\n"
        "        if self.activation == 'relu_n1_to_1':\n"
        "            return torch.clamp(x, min=-1.0, max=1.0)\n"
        "        if self.activation == 'relu_0_to_1':\n"
        "            return torch.clamp(x, min=0.0, max=1.0)\n"
        "        if self.activation == 'tanh':\n"
        "            return torch.tanh(x)\n"
        "        if self.activation == 'sigmoid':\n"
        "            return torch.sigmoid(x)\n"
        "        if self.activation == 'silu':\n"
        "            return torch.mul(x, torch.sigmoid(x))\n"
        "        return x\n\n"
    ) if has_conv_blocks else ""
    has_sequence_lstm_blocks = any(
        str(op.op_type) in {"UNIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_LSTM"}
        for op in model_ir.operators
    )
    has_sequence_rnn_blocks = any(
        str(op.op_type) == "UNIDIRECTIONAL_SEQUENCE_RNN"
        for op in model_ir.operators
    )
    sequence_rnn_helper_source = (
        "class _SequenceRNNBlock(torch.nn.Module):\n"
        "    def __init__(self, *, input_size: int, hidden_size: int, activation: str) -> None:\n"
        "        super().__init__()\n"
        "        activation_key = str(activation).strip().lower()\n"
        "        if activation_key not in {'tanh', 'relu'}:\n"
        "            raise ValueError(f'Unsupported RNN activation for native codegen: {activation}')\n"
        "        self.rnn = torch.nn.RNN(\n"
        "            input_size=int(input_size),\n"
        "            hidden_size=int(hidden_size),\n"
        "            num_layers=1,\n"
        "            nonlinearity=activation_key,\n"
        "            bias=True,\n"
        "            batch_first=False,\n"
        "            bidirectional=False,\n"
        "        )\n\n"
        "    def forward(\n"
        "        self,\n"
        "        x: torch.Tensor,\n"
        "        h0: Optional[torch.Tensor] = None,\n"
        "    ) -> torch.Tensor:\n"
        "        state: Optional[torch.Tensor] = None\n"
        "        if h0 is not None:\n"
        "            state = torch.unsqueeze(h0, dim=0)\n"
        "        y, _ = self.rnn(x, state) if state is not None else self.rnn(x)\n"
        "        return y\n\n"
    ) if has_sequence_rnn_blocks else ""
    sequence_lstm_helper_source = (
        "class _SequenceLSTMBlock(torch.nn.Module):\n"
        "    def __init__(self, *, input_size: int, hidden_size: int, sequence_length: Optional[int], bidirectional: bool, merge_outputs: bool) -> None:\n"
        "        super().__init__()\n"
        "        self.hidden_size = int(hidden_size)\n"
        "        self.sequence_length = int(sequence_length) if sequence_length is not None and int(sequence_length) > 0 else None\n"
        "        self.bidirectional = bool(bidirectional)\n"
        "        self.merge_outputs = bool(merge_outputs)\n"
        "        self.lstm = torch.nn.LSTM(\n"
        "            input_size=int(input_size),\n"
        "            hidden_size=int(hidden_size),\n"
        "            num_layers=1,\n"
        "            bias=True,\n"
        "            batch_first=False,\n"
        "            bidirectional=bool(bidirectional),\n"
        "        )\n\n"
        "    def forward(\n"
        "        self,\n"
        "        x: torch.Tensor,\n"
        "        fw_h0: Optional[torch.Tensor] = None,\n"
        "        fw_c0: Optional[torch.Tensor] = None,\n"
        "        bw_h0: Optional[torch.Tensor] = None,\n"
        "        bw_c0: Optional[torch.Tensor] = None,\n"
        "    ) -> torch.Tensor:\n"
        "        if self.sequence_length is not None:\n"
        "            x = torch.reshape(x, (self.sequence_length, x.shape[1], self.lstm.input_size))\n"
        "        else:\n"
        "            x = torch.reshape(x, (x.shape[0], x.shape[1], self.lstm.input_size))\n"
        "        if not self.bidirectional:\n"
        "            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None\n"
        "            if fw_h0 is not None and fw_c0 is not None:\n"
        "                state = (torch.unsqueeze(fw_h0, dim=0), torch.unsqueeze(fw_c0, dim=0))\n"
        "            y, _ = self.lstm(x, state) if state is not None else self.lstm(x)\n"
        "            return y\n"
        "        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None\n"
        "        if fw_h0 is not None or fw_c0 is not None or bw_h0 is not None or bw_c0 is not None:\n"
        "            batch_size = x.shape[1]\n"
        "            zeros = torch.zeros((batch_size, self.hidden_size), dtype=x.dtype, device=x.device)\n"
        "            fw_h = fw_h0 if fw_h0 is not None else zeros\n"
        "            fw_c = fw_c0 if fw_c0 is not None else zeros\n"
        "            bw_h = bw_h0 if bw_h0 is not None else zeros\n"
        "            bw_c = bw_c0 if bw_c0 is not None else zeros\n"
        "            fw_h = torch.reshape(fw_h, (batch_size, self.hidden_size))\n"
        "            fw_c = torch.reshape(fw_c, (batch_size, self.hidden_size))\n"
        "            bw_h = torch.reshape(bw_h, (batch_size, self.hidden_size))\n"
        "            bw_c = torch.reshape(bw_c, (batch_size, self.hidden_size))\n"
        "            state = (\n"
        "                torch.stack([fw_h, bw_h], dim=0),\n"
        "                torch.stack([fw_c, bw_c], dim=0),\n"
        "            )\n"
        "        y, _ = self.lstm(x, state) if state is not None else self.lstm(x)\n"
        "        if self.merge_outputs:\n"
        "            return y\n"
        "        return torch.stack([y[:, :, : self.hidden_size], y[:, :, self.hidden_size :]], dim=2)\n\n"
    ) if has_sequence_lstm_blocks else ""
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
        "_ONNX2TF_DISABLE_SYMBOLIC_SHAPE_TENSORS = False\n\n"
        "def _torch_dtype(dtype_name: str) -> torch.dtype:\n"
        "    key = str(dtype_name).upper()\n"
        "    if key not in _TORCH_DTYPE_BY_TFLITE_DTYPE:\n"
        "        raise RuntimeError(f'Unsupported dtype for PyTorch runtime: {dtype_name}')\n"
        "    return _TORCH_DTYPE_BY_TFLITE_DTYPE[key]\n\n"
        "def _module_device(module: Any) -> torch.device:\n"
        "    if torch.jit.is_scripting():\n"
        "        return torch.device('cpu')\n"
        "    for parameter in module.parameters():\n"
        "        return parameter.device\n"
        "    for buffer in module.buffers():\n"
        "        return buffer.device\n"
        "    return torch.device('cpu')\n\n"
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
        "def resolve_named_input_value(kwargs: Dict[str, Any], expected_name: str) -> Any:\n"
        "    return _resolve_named_input_value(kwargs, expected_name)\n\n"
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
        "def _torch_permute(value: torch.Tensor, perm: List[int]) -> torch.Tensor:\n"
        "    if len(perm) == 0:\n"
        "        return value\n"
        "    return torch.permute(value, perm).contiguous()\n\n"
        "def _can_broadcast_shapes(lhs: List[int], rhs: List[int]) -> bool:\n"
        "    lhs_index = len(lhs) - 1\n"
        "    rhs_index = len(rhs) - 1\n"
        "    while lhs_index >= 0 and rhs_index >= 0:\n"
        "        lhs_dim = int(lhs[lhs_index])\n"
        "        rhs_dim = int(rhs[rhs_index])\n"
        "        if lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:\n"
        "            return False\n"
        "        lhs_index -= 1\n"
        "        rhs_index -= 1\n"
        "    return True\n\n"
        "def _broadcast_shape(lhs: Sequence[int], rhs: Sequence[int]) -> Optional[List[int]]:\n"
        "    lhs_items = [int(v) for v in list(lhs)]\n"
        "    rhs_items = [int(v) for v in list(rhs)]\n"
        "    result_rev = torch.jit.annotate(List[int], [])\n"
        "    lhs_index = len(lhs_items) - 1\n"
        "    rhs_index = len(rhs_items) - 1\n"
        "    while lhs_index >= 0 or rhs_index >= 0:\n"
        "        lhs_dim = int(lhs_items[lhs_index]) if lhs_index >= 0 else 1\n"
        "        rhs_dim = int(rhs_items[rhs_index]) if rhs_index >= 0 else 1\n"
        "        if lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:\n"
        "            return None\n"
        "        result_rev.append(int(lhs_dim if rhs_dim == 1 else rhs_dim))\n"
        "        lhs_index -= 1\n"
        "        rhs_index -= 1\n"
        "    result = torch.jit.annotate(List[int], [])\n"
        "    reverse_index = len(result_rev) - 1\n"
        "    while reverse_index >= 0:\n"
        "        result.append(int(result_rev[reverse_index]))\n"
        "        reverse_index -= 1\n"
        "    return result\n\n"
        "def _align_tensor_to_target_shape(value: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    if target_shape is None:\n"
        "        return value\n"
        "    has_actual_shape, actual_shape = _tensor_static_shape_list(value)\n"
        "    if not has_actual_shape:\n"
        "        return value\n"
        "    target = [int(v) for v in list(target_shape)]\n"
        "    if _shape_matches_target_relaxed(actual_shape, target):\n"
        "        return value\n"
        "    perm = _perm_cl_to_cf(value.ndim)\n"
        "    permuted_shape = _permute_shape(actual_shape, perm) if perm is not None else None\n"
        "    if permuted_shape is not None and _shape_matches_target_relaxed(permuted_shape, target):\n"
        "        return _torch_permute(value, perm)\n"
        "    perm_inv = _perm_cf_to_cl(value.ndim)\n"
        "    permuted_shape_inv = _permute_shape(actual_shape, perm_inv) if perm_inv is not None else None\n"
        "    if permuted_shape_inv is not None and _shape_matches_target_relaxed(permuted_shape_inv, target):\n"
        "        return _torch_permute(value, perm_inv)\n"
        "    if len(actual_shape) == len(target):\n"
        "        if all(int(dim) == 1 for dim in list(target)) and any(int(actual_dim) > 1 for actual_dim in list(actual_shape)):\n"
        "            return value\n"
        "        can_narrow = True\n"
        "        has_mismatch = False\n"
        "        for dim_idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target)):\n"
        "            if int(target_dim) <= 0 or int(actual_dim) < int(target_dim):\n"
        "                can_narrow = False\n"
        "                break\n"
        "            if int(actual_dim) != int(target_dim):\n"
        "                has_mismatch = True\n"
        "                if int(dim_idx) == 0:\n"
        "                    can_narrow = False\n"
        "                    break\n"
        "        if can_narrow and has_mismatch:\n"
        "            narrowed = value\n"
        "            for dim_idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target)):\n"
        "                if int(actual_dim) > int(target_dim):\n"
        "                    narrowed = torch.narrow(narrowed, int(dim_idx), 0, int(target_dim))\n"
        "            return narrowed\n"
        "    return value\n\n"
        "@torch.jit.ignore\n"
        "def _align_scatter_nd_updates_eager(updates: torch.Tensor, expected_shape: Sequence[int]) -> torch.Tensor:\n"
        "    has_actual_shape, actual_shape = _tensor_static_shape_list(updates)\n"
        "    if not has_actual_shape:\n"
        "        return updates\n"
        "    expected = []\n"
        "    for dim in list(expected_shape):\n"
        "        if not isinstance(dim, int):\n"
        "            return updates\n"
        "        expected.append(int(dim))\n"
        "    try:\n"
        "        if list(torch.broadcast_shapes(tuple(actual_shape), tuple(expected))) == expected:\n"
        "            return updates\n"
        "    except Exception:\n"
        "        pass\n"
        "    perm = _perm_cf_to_cl(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    perm = _perm_cl_to_cf(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    if updates.ndim <= 5:\n"
        "        import itertools\n"
        "        for generic_perm in itertools.permutations(range(updates.ndim)):\n"
        "            if list(generic_perm) == list(range(updates.ndim)):\n"
        "                continue\n"
        "            generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "            if _permute_shape(actual_shape, generic_perm_list) == expected:\n"
        "                return _torch_permute(updates, generic_perm_list)\n"
        "    return updates\n\n"
        "def _align_scatter_nd_updates(updates: torch.Tensor, expected_shape: Sequence[int]) -> torch.Tensor:\n"
        "    has_actual_shape, actual_shape = _tensor_static_shape_list(updates)\n"
        "    expected = torch.jit.annotate(List[int], [])\n"
        "    for dim in list(expected_shape):\n"
        "        if not isinstance(dim, int):\n"
        "            return updates\n"
        "        expected.append(int(dim))\n"
        "    if not has_actual_shape:\n"
        "        return updates\n"
        "    if actual_shape == expected:\n"
        "        return updates\n"
        "    broadcast_shape = _broadcast_shape(actual_shape, expected)\n"
        "    if broadcast_shape is not None and [int(v) for v in list(broadcast_shape)] == expected:\n"
        "        return updates\n"
        "    perm = _perm_cf_to_cl(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    perm = _perm_cl_to_cf(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    if torch.jit.is_scripting():\n"
        "        return updates\n"
        "    return _align_scatter_nd_updates_eager(updates, expected)\n\n"
        "def _matches_target_except_axis(actual_shape: Sequence[int], target_shape: Sequence[int], axis: int) -> bool:\n"
        "    if len(list(actual_shape)) != len(list(target_shape)):\n"
        "        return False\n"
        "    for idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target_shape)):\n"
        "        if int(idx) == int(axis):\n"
        "            continue\n"
        "        if int(target_dim) <= 1:\n"
        "            continue\n"
        "        if int(actual_dim) != int(target_dim):\n"
        "            return False\n"
        "    return True\n\n"
        "def _optional_static_shape_list(shape: Optional[Sequence[int]]) -> Tuple[bool, List[int]]:\n"
        "    if shape is None:\n"
        "        return False, []\n"
        "    values = torch.jit.annotate(List[int], [])\n"
        "    for dim in list(shape):\n"
        "        if not isinstance(dim, int):\n"
        "            return False, []\n"
        "        values.append(int(dim))\n"
        "    return True, values\n\n"
        "def _shape_matches_target_relaxed(actual_shape: Sequence[int], target_shape: Sequence[int]) -> bool:\n"
        "    if len(list(actual_shape)) != len(list(target_shape)):\n"
        "        return False\n"
        "    for actual_dim, target_dim in zip(actual_shape, target_shape):\n"
        "        if int(target_dim) <= 0:\n"
        "            continue\n"
        "        if int(actual_dim) != int(target_dim):\n"
        "            return False\n"
        "    return True\n\n"
        "def _target_shape_has_concrete_dim(target_shape: Optional[Sequence[int]]) -> bool:\n"
        "    if target_shape is None:\n"
        "        return False\n"
        "    for dim in list(target_shape):\n"
        "        if int(dim) > 0:\n"
        "            return True\n"
        "    return False\n\n"
        "def _tensor_static_shape_list(value: torch.Tensor) -> Tuple[bool, List[int]]:\n"
        "    values = torch.jit.annotate(List[int], [])\n"
        "    if torch.jit.is_scripting():\n"
        "        for dim in list(value.shape):\n"
        "            values.append(int(dim))\n"
        "        return True, values\n"
        "    for dim in list(value.shape):\n"
        "        if not isinstance(dim, int):\n"
        "            return False, []\n"
        "        values.append(int(dim))\n"
        "    return True, values\n\n"
        "@torch.jit.ignore\n"
        "def _tensor_shape_list_eager(value: torch.Tensor) -> List[int]:\n"
        "    return list(value.shape)\n\n"
        "def _tensor_shape_list(value: torch.Tensor) -> List[int]:\n"
        "    if torch.jit.is_scripting():\n"
        "        result = torch.jit.annotate(List[int], [])\n"
        "        for dim in list(value.shape):\n"
        "            result.append(int(dim))\n"
        "        return result\n"
        "    return _tensor_shape_list_eager(value)\n\n"
        "@torch.jit.ignore\n"
        "def _align_binary_inputs_eager(x: torch.Tensor, y: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    try:\n"
        "        broadcast_shape = list(torch.broadcast_shapes(tuple(int(v) for v in x.shape), tuple(int(v) for v in y.shape)))\n"
        "        if target is None or _shape_matches_target_relaxed(broadcast_shape, target):\n"
            "            return x, y\n"
        "    except Exception:\n"
            "        pass\n"
        "    perm = _perm_cl_to_cf(x.ndim)\n"
        "    if perm is None:\n"
        "        return x, y\n"
        "    x_shape = [int(v) for v in list(x.shape)]\n"
        "    y_shape = [int(v) for v in list(y.shape)]\n"
        "    if _permute_shape(y_shape, perm) == x_shape:\n"
        "        return _align_binary_inputs_to_anchor(x, _torch_permute(y, perm), target_shape)\n"
        "    if _permute_shape(x_shape, perm) == y_shape:\n"
        "        return _align_binary_inputs_to_anchor(_torch_permute(x, perm), y, target_shape)\n"
        "    if target is not None:\n"
        "        if _shape_matches_target_relaxed(_permute_shape(y_shape, perm), target):\n"
        "            return _align_binary_inputs_to_anchor(x, _torch_permute(y, perm), target_shape)\n"
        "        if _shape_matches_target_relaxed(_permute_shape(x_shape, perm), target):\n"
        "            return _align_binary_inputs_to_anchor(_torch_permute(x, perm), y, target_shape)\n"
        "        if x.ndim <= 5:\n"
        "            import itertools\n"
        "            for generic_perm in itertools.permutations(range(x.ndim)):\n"
        "                if list(generic_perm) == list(range(x.ndim)):\n"
        "                    continue\n"
        "                generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "                permuted_y_shape = _permute_shape(y_shape, generic_perm_list)\n"
        "                if permuted_y_shape is not None:\n"
        "                    try:\n"
        "                        broadcast_shape = list(torch.broadcast_shapes(tuple(permuted_y_shape), tuple(x_shape)))\n"
        "                        if _shape_matches_target_relaxed(broadcast_shape, target):\n"
        "                            return _align_binary_inputs_to_anchor(x, _torch_permute(y, generic_perm_list), target_shape)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                permuted_x_shape = _permute_shape(x_shape, generic_perm_list)\n"
        "                if permuted_x_shape is not None:\n"
        "                    try:\n"
        "                        broadcast_shape = list(torch.broadcast_shapes(tuple(permuted_x_shape), tuple(y_shape)))\n"
        "                        if _shape_matches_target_relaxed(broadcast_shape, target):\n"
        "                            return _align_binary_inputs_to_anchor(_torch_permute(x, generic_perm_list), y, target_shape)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "    if x.ndim <= 5:\n"
        "        import itertools\n"
        "        for generic_perm in itertools.permutations(range(x.ndim)):\n"
        "            if list(generic_perm) == list(range(x.ndim)):\n"
        "                continue\n"
        "            generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "            permuted_y_shape = _permute_shape(y_shape, generic_perm_list)\n"
        "            if permuted_y_shape is not None:\n"
        "                try:\n"
        "                    torch.broadcast_shapes(tuple(permuted_y_shape), tuple(x_shape))\n"
        "                    return x, _torch_permute(y, generic_perm_list)\n"
        "                except Exception:\n"
        "                    pass\n"
        "                if target is not None:\n"
        "                    try:\n"
        "                        torch.broadcast_shapes(tuple(permuted_y_shape), tuple(target))\n"
        "                        return x, _torch_permute(y, generic_perm_list)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "            permuted_x_shape = _permute_shape(x_shape, generic_perm_list)\n"
        "            if permuted_x_shape is not None:\n"
        "                try:\n"
        "                    torch.broadcast_shapes(tuple(permuted_x_shape), tuple(y_shape))\n"
        "                    return _torch_permute(x, generic_perm_list), y\n"
        "                except Exception:\n"
        "                    pass\n"
        "                if target is not None:\n"
        "                    try:\n"
        "                        torch.broadcast_shapes(tuple(permuted_x_shape), tuple(target))\n"
        "                        return _torch_permute(x, generic_perm_list), y\n"
        "                    except Exception:\n"
        "                        pass\n"
        "    return x, y\n\n"
        "def _align_binary_inputs(x: torch.Tensor, y: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    has_target, target = _optional_static_shape_list(target_shape)\n"
        "    if x.ndim != y.ndim:\n"
        "        return x, y\n"
        "    has_x_shape, x_shape = _tensor_static_shape_list(x)\n"
        "    has_y_shape, y_shape = _tensor_static_shape_list(y)\n"
        "    if not has_x_shape or not has_y_shape:\n"
        "        return x, y\n"
        "    if x_shape == y_shape:\n"
        "        return x, y\n"
        "    broadcast_shape = _broadcast_shape(x_shape, y_shape)\n"
        "    if broadcast_shape is not None:\n"
        "        x_is_scalar_seed = all(int(dim) == 1 for dim in list(x_shape))\n"
        "        y_is_scalar_seed = all(int(dim) == 1 for dim in list(y_shape))\n"
        "        if x_is_scalar_seed or y_is_scalar_seed:\n"
        "            return x, y\n"
        "        if not has_target or _shape_matches_target_relaxed(broadcast_shape, target):\n"
        "            return x, y\n"
        "        if torch.jit.is_scripting():\n"
        "            return x, y\n"
        "        return _align_binary_inputs_eager(x, y, target_shape)\n"
        "    if _can_broadcast_shapes(x_shape, y_shape):\n"
            "        if not has_target:\n"
            "            return x, y\n"
        "        broadcast_shape = _broadcast_shape(x_shape, y_shape)\n"
        "        if broadcast_shape is not None and _shape_matches_target_relaxed(broadcast_shape, target):\n"
        "                return x, y\n"
        "        if torch.jit.is_scripting():\n"
        "            return x, y\n"
        "        return _align_binary_inputs_eager(x, y, target_shape)\n"
        "    perm = _perm_cl_to_cf(x.ndim)\n"
        "    if perm is None:\n"
        "        return x, y\n"
        "    if _permute_shape(y_shape, perm) == x_shape:\n"
        "        return _align_binary_inputs_to_anchor(x, _torch_permute(y, perm), target_shape)\n"
        "    if _permute_shape(x_shape, perm) == y_shape:\n"
        "        return _align_binary_inputs_to_anchor(_torch_permute(x, perm), y, target_shape)\n"
        "    if has_target:\n"
        "        if _shape_matches_target_relaxed(_permute_shape(y_shape, perm), target):\n"
        "            return _align_binary_inputs_to_anchor(x, _torch_permute(y, perm), target_shape)\n"
        "        if _shape_matches_target_relaxed(_permute_shape(x_shape, perm), target):\n"
        "            return _align_binary_inputs_to_anchor(_torch_permute(x, perm), y, target_shape)\n"
        "    if torch.jit.is_scripting():\n"
        "        return x, y\n"
        "    return _align_binary_inputs_eager(x, y, target_shape)\n\n"
        "@torch.jit.ignore\n"
        "def _align_binary_inputs_to_anchor_eager(anchor: torch.Tensor, other: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    anchor_shape = [int(v) for v in list(anchor.shape)]\n"
        "    other_shape = [int(v) for v in list(other.shape)]\n"
        "    perm = _perm_cl_to_cf(other.ndim)\n"
        "    if perm is not None and _permute_shape(other_shape, perm) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm)\n"
        "    perm_inv = _perm_cf_to_cl(other.ndim)\n"
        "    if perm_inv is not None and _permute_shape(other_shape, perm_inv) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm_inv)\n"
        "    if target is not None and other.ndim <= 5:\n"
        "        import itertools\n"
        "        for generic_perm in itertools.permutations(range(other.ndim)):\n"
        "            if list(generic_perm) == list(range(other.ndim)):\n"
        "                continue\n"
        "            generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "            permuted_shape = _permute_shape(other_shape, generic_perm_list)\n"
        "            if permuted_shape is None:\n"
        "                continue\n"
        "            try:\n"
        "                broadcast_shape = list(torch.broadcast_shapes(tuple(anchor_shape), tuple(permuted_shape)))\n"
        "                if _target_shape_has_concrete_dim(target) and _shape_matches_target_relaxed(broadcast_shape, target):\n"
        "                    return anchor, _torch_permute(other, generic_perm_list)\n"
        "            except Exception:\n"
        "                pass\n"
        "    if other.ndim <= 5:\n"
        "        import itertools\n"
        "        for generic_perm in itertools.permutations(range(other.ndim)):\n"
        "            if list(generic_perm) == list(range(other.ndim)):\n"
        "                continue\n"
        "            generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "            permuted_shape = _permute_shape(other_shape, generic_perm_list)\n"
        "            if permuted_shape == anchor_shape:\n"
        "                return anchor, _torch_permute(other, generic_perm_list)\n"
        "            if permuted_shape is not None:\n"
        "                try:\n"
        "                    broadcast_shape = list(torch.broadcast_shapes(tuple(anchor_shape), tuple(permuted_shape)))\n"
        "                    if broadcast_shape == anchor_shape or (target is not None and _target_shape_has_concrete_dim(target) and _shape_matches_target_relaxed(broadcast_shape, target)):\n"
        "                        return anchor, _torch_permute(other, generic_perm_list)\n"
        "                except Exception:\n"
        "                    pass\n"
        "    return anchor, other\n\n"
        "def _align_binary_inputs_to_anchor(anchor: torch.Tensor, other: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    has_target, target = _optional_static_shape_list(target_shape)\n"
        "    if anchor.ndim != other.ndim:\n"
        "        return anchor, other\n"
        "    has_anchor_shape, anchor_shape = _tensor_static_shape_list(anchor)\n"
        "    has_other_shape, other_shape = _tensor_static_shape_list(other)\n"
        "    if not has_anchor_shape or not has_other_shape:\n"
        "        return anchor, other\n"
        "    if anchor_shape == other_shape:\n"
        "        return anchor, other\n"
        "    broadcast_shape = _broadcast_shape(anchor_shape, other_shape)\n"
        "    if broadcast_shape is not None:\n"
        "        if broadcast_shape == anchor_shape:\n"
            "            return anchor, other\n"
        "        if not has_target:\n"
            "            return anchor, other\n"
        "        if _target_shape_has_concrete_dim(target) and _shape_matches_target_relaxed(broadcast_shape, target):\n"
            "            return anchor, other\n"
        "        if torch.jit.is_scripting():\n"
            "            return anchor, other\n"
        "        return _align_binary_inputs_to_anchor_eager(anchor, other, target_shape)\n"
        "    perm = _perm_cl_to_cf(other.ndim)\n"
        "    if perm is not None and _permute_shape(other_shape, perm) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm)\n"
        "    perm_inv = _perm_cf_to_cl(other.ndim)\n"
        "    if perm_inv is not None and _permute_shape(other_shape, perm_inv) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm_inv)\n"
        "    if has_target and _can_broadcast_shapes(anchor_shape, other_shape):\n"
        "        broadcast_shape = _broadcast_shape(anchor_shape, other_shape)\n"
        "        if broadcast_shape is not None and (broadcast_shape == anchor_shape or (_target_shape_has_concrete_dim(target) and _shape_matches_target_relaxed(broadcast_shape, target))):\n"
        "            return anchor, other\n"
        "        if torch.jit.is_scripting():\n"
        "            return anchor, other\n"
        "        return _align_binary_inputs_to_anchor_eager(anchor, other, target_shape)\n"
        "    if torch.jit.is_scripting():\n"
        "        return anchor, other\n"
        "    return _align_binary_inputs_to_anchor_eager(anchor, other, target_shape)\n\n"
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
        "def _apply_cumsum(x: torch.Tensor, *, axis: int, exclusive: bool, reverse: bool) -> torch.Tensor:\n"
        "    dim = _normalize_dim(int(axis), x.ndim)\n"
        "    y = torch.flip(x, dims=[dim]) if reverse else x\n"
        "    y = torch.cumsum(y, dim=dim)\n"
        "    if exclusive:\n"
        "        axis_size = int(y.shape[dim])\n"
        "        if axis_size > 0:\n"
        "            zeros = torch.zeros_like(torch.narrow(y, dim, 0, 1))\n"
        "            prefix = torch.narrow(y, dim, 0, max(axis_size - 1, 0))\n"
        "            y = torch.cat([zeros, prefix], dim=dim)\n"
        "    if reverse:\n"
        "        y = torch.flip(y, dims=[dim])\n"
        "    return y\n\n"
        "def _shape_list(value: Any) -> List[int]:\n"
        "    if isinstance(value, torch.Tensor):\n"
        "        flat = value.to(dtype=torch.int64).reshape(-1)\n"
        "        if torch.jit.is_scripting():\n"
        "            result = torch.jit.annotate(List[int], [])\n"
        "            for idx in range(int(flat.numel())):\n"
        "                result.append(int(flat[idx].item()))\n"
        "            return result\n"
        "        eager_result = []\n"
        "        for idx in range(int(flat.numel())):\n"
        "            eager_result.append(flat[idx])\n"
        "        return eager_result\n"
        "    if torch.jit.isinstance(value, List[int]):\n"
        "        result = torch.jit.annotate(List[int], [])\n"
        "        for item in value:\n"
        "            result.append(int(item))\n"
        "        return result\n"
        "    raise RuntimeError('Unsupported shape spec type for _shape_list')\n\n"
        "def _reshape_gather_output(gathered: torch.Tensor, params: torch.Tensor, indices_shape_spec: Sequence[int], *, axis: int) -> torch.Tensor:\n"
        "    resolved_axis = _normalize_dim(int(axis), params.ndim)\n"
        "    target_shape = torch.jit.annotate(List[int], [])\n"
        "    for dim_index in range(int(resolved_axis)):\n"
        "        target_shape.append(params.shape[dim_index])\n"
        "    for dim_value in indices_shape_spec:\n"
        "        target_shape.append(dim_value)\n"
        "    for dim_index in range(int(resolved_axis) + 1, int(params.ndim)):\n"
        "        target_shape.append(params.shape[dim_index])\n"
        "    return torch.reshape(gathered, target_shape)\n\n"
        "@torch.jit.ignore\n"
        "def _apply_tile_eager(x: torch.Tensor, multiples: Any) -> torch.Tensor:\n"
        "    repeats = list(_shape_list(multiples))\n"
        "    target_rank = len(repeats)\n"
        "    if target_rank < int(x.ndim):\n"
        "        repeats = ([1] * (int(x.ndim) - target_rank)) + repeats\n"
        "        target_rank = len(repeats)\n"
        "    y = x\n"
        "    if target_rank > int(y.ndim):\n"
        "        reshape_shape = [1] * (target_rank - int(y.ndim))\n"
        "        for axis in range(int(y.ndim)):\n"
        "            reshape_shape.append(y.shape[axis])\n"
        "        y = torch.reshape(y, reshape_shape)\n"
        "    for dim_index in range(target_rank - 1, -1, -1):\n"
        "        repeat = repeats[dim_index]\n"
        "        y = torch.unsqueeze(y, dim_index)\n"
        "        expanded_shape = []\n"
        "        for axis in range(int(y.ndim)):\n"
        "            if axis == dim_index:\n"
        "                expanded_shape.append(repeat)\n"
        "                continue\n"
        "            expanded_shape.append(y.shape[axis])\n"
        "        y = y.expand(expanded_shape)\n"
        "        merged_shape = []\n"
        "        for axis in range(int(y.ndim)):\n"
        "            if axis == dim_index:\n"
        "                merged_shape.append(y.shape[axis] * y.shape[axis + 1])\n"
                "                continue\n"
        "            if axis == dim_index + 1:\n"
        "                continue\n"
        "            merged_shape.append(y.shape[axis])\n"
        "        y = torch.reshape(y, merged_shape)\n"
        "    return y\n\n"
        "def _apply_tile(x: torch.Tensor, multiples: Any) -> torch.Tensor:\n"
        "    if torch.jit.is_scripting():\n"
        "        return torch.tile(x, _shape_list(multiples))\n"
        "    return _apply_tile_eager(x, multiples)\n\n"
        "def _apply_random_standard_normal(shape_spec: Any, *, dtype: torch.dtype, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:\n"
        "    shape = _shape_list(shape_spec)\n"
        "    if seed is None:\n"
        "        return torch.randn(shape, dtype=dtype, device=device)\n"
        "    if torch.jit.is_scripting():\n"
        "        torch.manual_seed(int(seed))\n"
        "        return torch.randn(shape, dtype=dtype, device=device)\n"
        "    generator = torch.Generator(device=device)\n"
        "    generator.manual_seed(int(seed))\n"
        "    return torch.randn(shape, dtype=dtype, device=device, generator=generator)\n\n"
        "def _shape_tensor(input_tensor: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:\n"
        "    values = torch.jit.annotate(List[int], [])\n"
        "    for idx in range(int(input_tensor.ndim)):\n"
        "        values.append(input_tensor.shape[idx])\n"
        "    if len(values) == 0:\n"
        "        return torch.empty((0,), dtype=dtype, device=device)\n"
        "    return torch.tensor(values, dtype=dtype, device=device)\n\n"
        "def _resolve_reshape_shape(shape_spec: Any, input_tensor: torch.Tensor, *, allow_zero: bool) -> List[int]:\n"
        "    raw_shape = _shape_list(shape_spec)\n"
        "    resolved: List[int] = []\n"
        "    raw_minus_one_count = 0\n"
        "    for raw_dim in raw_shape:\n"
        "        if int(raw_dim) == -1:\n"
        "            raw_minus_one_count += 1\n"
        "    infer_count = 0\n"
        "    input_rank = int(input_tensor.ndim)\n"
        "    for dim_index, raw_dim in enumerate(raw_shape):\n"
        "        dim_value = int(raw_dim)\n"
        "        if dim_index < input_rank and dim_value <= 0:\n"
        "            if dim_value == -1 and raw_minus_one_count > 1:\n"
        "                resolved.append(input_tensor.shape[dim_index])\n"
        "                continue\n"
        "            if dim_value == 0 and not allow_zero:\n"
        "                resolved.append(input_tensor.shape[dim_index])\n"
        "                continue\n"
        "        if dim_value == 0 and not allow_zero and dim_index < input_rank:\n"
        "            resolved.append(input_tensor.shape[dim_index])\n"
        "            continue\n"
        "        if dim_value == -1:\n"
        "            infer_count += 1\n"
        "            if infer_count > 1:\n"
        "                raise RuntimeError(f'Multiple -1 values are not allowed in reshape spec: {raw_shape}')\n"
        "        resolved.append(dim_value)\n"
        "    if infer_count == 1:\n"
        "        infer_index = -1\n"
        "        known_product = 1\n"
        "        for dim_index, dim_value in enumerate(resolved):\n"
        "            if int(dim_value) == -1:\n"
        "                infer_index = int(dim_index)\n"
        "                continue\n"
        "            known_product *= dim_value\n"
        "        if infer_index >= 0:\n"
        "            resolved[infer_index] = input_tensor.numel() // known_product\n"
        "    return resolved\n\n"
        "def _resolve_reshape_shape_tensor(shape_spec: torch.Tensor, input_tensor: torch.Tensor, *, allow_zero: bool) -> torch.Tensor:\n"
        "    flat = shape_spec.to(dtype=torch.int64, device=input_tensor.device).reshape(-1)\n"
        "    if allow_zero:\n"
        "        return flat\n"
        "    resolved = torch.jit.annotate(List[torch.Tensor], [])\n"
        "    input_rank = int(input_tensor.ndim)\n"
        "    for dim_index in range(int(flat.numel())):\n"
        "        dim_value = flat[dim_index]\n"
        "        if dim_index < input_rank:\n"
        "            input_dim = torch.scalar_tensor(input_tensor.shape[dim_index], dtype=flat.dtype, device=flat.device)\n"
        "            resolved.append(torch.where(dim_value == 0, input_dim, dim_value))\n"
        "            continue\n"
        "        resolved.append(dim_value)\n"
        "    if len(resolved) == 0:\n"
        "        return flat\n"
        "    return torch.stack(resolved)\n\n"
        "def _to_torch_pad_arg(paddings: torch.Tensor) -> List[int]:\n"
        "    pads_tensor = paddings.to(dtype=torch.int64).reshape(-1, 2)\n"
        "    torch_pad: List[int] = []\n"
        "    for row_index in range(int(pads_tensor.shape[0]) - 1, -1, -1):\n"
        "        torch_pad.extend([\n"
        "            int(pads_tensor[row_index, 0].item()),\n"
        "            int(pads_tensor[row_index, 1].item()),\n"
        "        ])\n"
        "    while len(torch_pad) >= 2 and int(torch_pad[-2]) == 0 and int(torch_pad[-1]) == 0:\n"
        "        torch_pad = torch_pad[:-2]\n"
        "    return torch_pad\n\n"
        "def _apply_pad_nd(x: torch.Tensor, paddings: torch.Tensor, *, mode: str, value: float = 0.0) -> torch.Tensor:\n"
        "    pads_tensor = paddings.to(dtype=torch.int64).reshape(-1, 2)\n"
        "    if (not torch.jit.is_scripting()) and str(mode) == 'constant':\n"
        "        rank = int(x.ndim)\n"
        "        pad_rows = int(pads_tensor.shape[0])\n"
        "        if pad_rows < rank:\n"
        "            prefix_rows = torch.zeros([rank - pad_rows, 2], dtype=pads_tensor.dtype, device=pads_tensor.device)\n"
        "            pads_tensor = torch.cat([prefix_rows, pads_tensor], dim=0)\n"
        "        elif pad_rows > rank:\n"
        "            pads_tensor = pads_tensor[(pad_rows - rank):, :]\n"
        "        y = x\n"
        "        for axis in range(rank):\n"
        "            before = torch.reshape(pads_tensor[axis, 0], [1]).to(dtype=torch.int64)\n"
        "            after = torch.reshape(pads_tensor[axis, 1], [1]).to(dtype=torch.int64)\n"
        "            zero_shape = list(y.shape)\n"
        "            zero_shape[axis] = 1\n"
        "            zero_slice = torch.full(zero_shape, float(value), dtype=y.dtype, device=y.device)\n"
        "            pad_before = torch.repeat_interleave(zero_slice, before, dim=axis)\n"
        "            pad_after = torch.repeat_interleave(zero_slice, after, dim=axis)\n"
        "            y = torch.cat([pad_before, y, pad_after], dim=axis)\n"
        "        return y\n"
        "    pad_pairs = torch.jit.annotate(List[List[int]], [])\n"
        "    for row_index in range(int(pads_tensor.shape[0])):\n"
        "        pad_pairs.append([\n"
        "            int(pads_tensor[row_index, 0].item()),\n"
        "            int(pads_tensor[row_index, 1].item()),\n"
        "        ])\n"
        "    rank = int(x.ndim)\n"
        "    if len(pad_pairs) < rank:\n"
        "        pad_pairs = ([[0, 0]] * (rank - len(pad_pairs))) + pad_pairs\n"
        "    elif len(pad_pairs) > rank:\n"
        "        pad_pairs = pad_pairs[-rank:]\n"
        "    non_zero_axes = [idx for idx, (before, after) in enumerate(pad_pairs) if int(before) != 0 or int(after) != 0]\n"
        "    if len(non_zero_axes) == 0:\n"
        "        return x\n"
        "    if mode != 'constant' and len(non_zero_axes) > 3:\n"
        "        raise RuntimeError(f'Non-constant pad supports at most 3 padded dims. mode={mode} padded_dims={len(non_zero_axes)}')\n"
        "    keep_axes = [idx for idx in range(rank) if idx not in non_zero_axes]\n"
        "    perm = keep_axes + non_zero_axes\n"
        "    permuted = _torch_permute(x, perm) if perm != list(range(rank)) else x\n"
        "    torch_pad: List[int] = []\n"
        "    for axis in reversed(non_zero_axes):\n"
        "        before, after = pad_pairs[axis]\n"
        "        torch_pad.extend([int(before), int(after)])\n"
        "    if mode == 'constant':\n"
        "        padded = F.pad(permuted, torch_pad, mode=mode, value=float(value))\n"
        "    else:\n"
        "        padded = F.pad(permuted, torch_pad, mode=mode)\n"
        "    if perm == list(range(rank)):\n"
        "        return padded\n"
        "    inverse_perm = [0] * rank\n"
        "    for permuted_axis, original_axis in enumerate(perm):\n"
        "        inverse_perm[int(original_axis)] = int(permuted_axis)\n"
        "    return _torch_permute(padded, inverse_perm)\n\n"
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
        "    if key == '' or key == 'NONE':\n"
        "        return x\n"
        "    if key == 'RELU':\n"
        "        return torch.relu(x)\n"
        "    if key == 'RELU6':\n"
        "        return torch.clamp(x, min=0.0, max=6.0)\n"
        "    if key == 'RELU_N1_TO_1':\n"
        "        return torch.clamp(x, min=-1.0, max=1.0)\n"
        "    if key == 'RELU_0_TO_1':\n"
        "        return torch.clamp(x, min=0.0, max=1.0)\n"
        "    if key == 'SILU':\n"
        "        return torch.mul(x, torch.sigmoid(x))\n"
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
        "def _apply_concat(values: Sequence[torch.Tensor], axis: int, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    if any(int(value.ndim) == 0 for value in values):\n"
        "        values = [value.reshape(1) if int(value.ndim) == 0 else value for value in values]\n"
        "    rank = int(values[0].ndim)\n"
        "    resolved_axis = _normalize_dim(int(axis), rank)\n"
        "    has_target, target_list = _optional_static_shape_list(target_shape)\n"
        "    if has_target and len(target_list) == rank:\n"
        "        candidate_targets: List[List[int]] = [target_list]\n"
        "        perm_cf_to_cl = _perm_cf_to_cl(rank)\n"
        "        if perm_cf_to_cl is not None:\n"
        "            alt_target = _permute_shape(target_list, perm_cf_to_cl)\n"
        "            if alt_target is not None:\n"
        "                already_present = False\n"
        "                for candidate_target in candidate_targets:\n"
        "                    if candidate_target == alt_target:\n"
        "                        already_present = True\n"
        "                        break\n"
        "                if not already_present:\n"
        "                    candidate_targets.append(alt_target)\n"
        "        perm_cl_to_cf = _perm_cl_to_cf(rank)\n"
        "        if perm_cl_to_cf is not None:\n"
        "            alt_target = _permute_shape(target_list, perm_cl_to_cf)\n"
        "            if alt_target is not None:\n"
        "                already_present = False\n"
        "                for candidate_target in candidate_targets:\n"
        "                    if candidate_target == alt_target:\n"
        "                        already_present = True\n"
        "                        break\n"
        "                if not already_present:\n"
        "                    candidate_targets.append(alt_target)\n"
        "        best_aligned_values: Optional[List[torch.Tensor]] = None\n"
        "        best_score: Optional[int] = None\n"
        "        for candidate_target in candidate_targets:\n"
        "            aligned_values = []\n"
        "            candidate_score = 0\n"
        "            feasible = True\n"
        "            for value in values:\n"
        "                has_actual, actual = _tensor_static_shape_list(value)\n"
        "                if not has_actual:\n"
        "                    aligned_values.append(value)\n"
        "                    continue\n"
        "                chosen = value\n"
        "                if actual != candidate_target:\n"
        "                    if _matches_target_except_axis(actual, candidate_target, resolved_axis):\n"
        "                        aligned_values.append(chosen)\n"
        "                        continue\n"
        "                    matched = False\n"
        "                    for perm in (_perm_cl_to_cf(value.ndim), _perm_cf_to_cl(value.ndim)):\n"
        "                        if perm is None:\n"
        "                            continue\n"
        "                        permuted_shape = _permute_shape(actual, perm)\n"
        "                        if _matches_target_except_axis(permuted_shape, candidate_target, resolved_axis):\n"
        "                            chosen = _torch_permute(value, perm)\n"
        "                            candidate_score += 1\n"
        "                            matched = True\n"
        "                            break\n"
        "                    if not matched:\n"
        "                        feasible = False\n"
        "                        break\n"
        "                aligned_values.append(chosen)\n"
        "            if feasible and (best_score is None or int(candidate_score) < int(best_score)):\n"
        "                best_aligned_values = aligned_values\n"
        "                best_score = int(candidate_score)\n"
        "        if best_aligned_values is not None:\n"
        "            values = best_aligned_values\n"
        "    y = torch.cat(list(values), dim=resolved_axis)\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_conv2d(module: torch.nn.Conv2d, x: torch.Tensor, target_shape: Optional[Sequence[int]], target_logical_layout: Optional[str], fused: str) -> torch.Tensor:\n"
        "    expected_in_channels = int(module.in_channels)\n"
        "    if x.ndim == 4 and int(x.shape[1]) != expected_in_channels and int(x.shape[-1]) == expected_in_channels:\n"
        "        x = x.permute(0, 3, 1, 2).contiguous()\n"
        "    y = module.forward(x)\n"
        "    if str(target_logical_layout).upper() == 'NHWC':\n"
        "        y = y.permute(0, 2, 3, 1).contiguous()\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_transpose_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], stride: List[int], padding: List[int], dilation: List[int], output_padding: List[int], groups: int, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], target_logical_layout: Optional[str], fused: str) -> torch.Tensor:\n"
        "    if x.ndim == 4 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):\n"
        "        x = x.permute(0, 3, 1, 2).contiguous()\n"
        "    raw = F.conv_transpose2d(x, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=int(groups), dilation=dilation)\n"
        "    target_h, target_w = _infer_spatial_shape_for_transposed_conv2d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)\n"
        "    y = raw[..., :target_h, :target_w]\n"
        "    if str(target_logical_layout).upper() == 'NHWC':\n"
        "        y = y.permute(0, 2, 3, 1).contiguous()\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_conv3d(module: torch.nn.Conv3d, x: torch.Tensor, target_shape: Optional[Sequence[int]], target_logical_layout: Optional[str], fused: str) -> torch.Tensor:\n"
        "    weight = module.weight\n"
        "    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[1]) and int(x.shape[-1]) == int(weight.shape[1]):\n"
        "        x = x.permute(0, 4, 1, 2, 3).contiguous()\n"
        "    y = module.forward(x)\n"
        "    if str(target_logical_layout).upper() == 'NDHWC':\n"
        "        y = y.permute(0, 2, 3, 4, 1).contiguous()\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_transpose_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], stride: List[int], padding: List[int], dilation: List[int], output_padding: List[int], groups: int, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], target_logical_layout: Optional[str], fused: str) -> torch.Tensor:\n"
        "    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):\n"
        "        x = x.permute(0, 4, 1, 2, 3).contiguous()\n"
        "    raw = F.conv_transpose3d(x, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=int(groups), dilation=dilation)\n"
        "    target_d, target_h, target_w = _infer_spatial_shape_for_transposed_conv3d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)\n"
        "    y = raw[..., :target_d, :target_h, :target_w]\n"
        "    if str(target_logical_layout).upper() == 'NDHWC':\n"
        "        y = y.permute(0, 2, 3, 4, 1).contiguous()\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_softmax(x: torch.Tensor, axis: Optional[int], beta: float, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    resolved_axis = _normalize_dim(int(axis), x.ndim) if axis is not None else -1\n"
        "    if beta != 1.0:\n"
        "        x = x * beta\n"
        "    y = torch.softmax(x, dim=resolved_axis)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "@torch.jit.ignore\n"
        "def _apply_gather_batched_eager(params: torch.Tensor, indices_i64: torch.Tensor, *, resolved_axis: int, resolved_batch_dims: int, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    params_flat = torch.flatten(params, 0, resolved_batch_dims - 1)\n"
        "    indices_flat = torch.flatten(indices_i64, 0, resolved_batch_dims - 1)\n"
        "    adjusted_axis = int(resolved_axis - resolved_batch_dims + 1)\n"
        "    prefix_shape = list(params_flat.shape[1:adjusted_axis])\n"
        "    suffix_shape = list(params_flat.shape[adjusted_axis + 1:])\n"
        "    index_shape = list(indices_flat.shape[1:])\n"
        "    input_view = params_flat\n"
        "    for _ in range(max(len(index_shape) - 1, 0)):\n"
        "        input_view = input_view.unsqueeze(adjusted_axis + 1)\n"
        "    input_expand_shape = list(params_flat.shape[:adjusted_axis + 1]) + list(index_shape[1:]) + suffix_shape\n"
        "    input_expanded = input_view.expand(input_expand_shape)\n"
        "    index_view_shape = [indices_flat.shape[0]] + ([1] * len(prefix_shape)) + index_shape + ([1] * len(suffix_shape))\n"
        "    index_view = indices_flat.reshape(index_view_shape)\n"
        "    index_expand_shape = [indices_flat.shape[0]] + prefix_shape + index_shape + suffix_shape\n"
        "    index_expanded = index_view.expand(index_expand_shape)\n"
        "    y = torch.gather(input_expanded, adjusted_axis, index_expanded)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_gather(params: torch.Tensor, indices: torch.Tensor, axis: int, batch_dims: int, target_shape: Optional[Sequence[int]], indices_name: str) -> torch.Tensor:\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    resolved_axis = _normalize_dim(int(axis), params.ndim)\n"
        "    if int(batch_dims) == 0 and int(resolved_axis) == 1 and str(indices_name).endswith('_crd_to_dcr_indices'):\n"
        "        return _align_tensor_to_target_shape(params, target_shape)\n"
        "    resolved_batch_dims = int(batch_dims)\n"
        "    if resolved_batch_dims < 0:\n"
        "        resolved_batch_dims += indices_i64.ndim\n"
        "    if resolved_batch_dims > 0:\n"
        "        if not torch.jit.is_scripting():\n"
        "            return _apply_gather_batched_eager(params, indices_i64, resolved_axis=resolved_axis, resolved_batch_dims=resolved_batch_dims, target_shape=target_shape)\n"
        "        leading_shape = torch.jit.annotate(List[int], [])\n"
        "        for dim_index in range(int(resolved_batch_dims)):\n"
        "            leading_shape.append(indices_i64.shape[dim_index])\n"
        "        params_flat = torch.flatten(params, 0, resolved_batch_dims - 1)\n"
        "        indices_flat = torch.flatten(indices_i64, 0, resolved_batch_dims - 1)\n"
        "        gathered_batches: List[torch.Tensor] = []\n"
        "        adjusted_axis = int(resolved_axis - resolved_batch_dims + 1)\n"
        "        for batch_index in range(params_flat.shape[0]):\n"
        "            batch_params = params_flat[batch_index]\n"
        "            batch_indices = indices_flat[batch_index]\n"
        "            flat_indices = batch_indices.reshape(-1)\n"
        "            batch_gathered = torch.index_select(batch_params, adjusted_axis - 1, flat_indices)\n"
        "            batch_gathered = _reshape_gather_output(batch_gathered, batch_params, list(batch_indices.shape), axis=adjusted_axis - 1)\n"
        "            gathered_batches.append(batch_gathered)\n"
        "        batch_result_shape = torch.jit.annotate(List[int], [])\n"
        "        for dim_value in leading_shape:\n"
        "            batch_result_shape.append(dim_value)\n"
        "        for dim_value in gathered_batches[0].shape:\n"
        "            batch_result_shape.append(dim_value)\n"
        "        y = torch.reshape(torch.stack(gathered_batches, dim=0), batch_result_shape)\n"
        "        return _align_tensor_to_target_shape(y, target_shape)\n"
        "    if indices_i64.ndim == 0:\n"
        "        y = torch.index_select(params, resolved_axis, indices_i64.reshape(1)).squeeze(resolved_axis)\n"
        "        return _align_tensor_to_target_shape(y, target_shape)\n"
        "    flat_indices = indices_i64.reshape(-1)\n"
        "    gathered = torch.index_select(params, resolved_axis, flat_indices)\n"
        "    y = _reshape_gather_output(gathered, params, list(indices_i64.shape), axis=resolved_axis)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_gather_elements(params: torch.Tensor, indices: torch.Tensor, axis_hint: int, axis_hint_cl: Optional[int], target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    def _gather_for_axis(axis: int) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "        resolved_axis = int(axis)\n"
        "        dim_size = int(params.shape[resolved_axis])\n"
        "        if dim_size <= 0:\n"
        "            safe_indices = torch.zeros_like(indices_i64)\n"
        "            gathered = torch.gather(params, dim=resolved_axis, index=safe_indices)\n"
        "            return gathered, torch.zeros((), dtype=torch.bool, device=indices_i64.device)\n"
        "        dim_size_tensor = torch.as_tensor(dim_size, dtype=torch.int64, device=indices_i64.device)\n"
        "        normalized_indices = torch.where(indices_i64 < 0, indices_i64 + dim_size_tensor, indices_i64)\n"
        "        axis_valid = torch.all((indices_i64 >= -dim_size_tensor) & (indices_i64 < dim_size_tensor))\n"
        "        safe_indices = torch.clamp(normalized_indices, min=0, max=dim_size - 1)\n"
        "        return torch.gather(params, dim=resolved_axis, index=safe_indices), axis_valid\n"
        "    gather_axis = int(axis_hint)\n"
        "    y, y_valid = _gather_for_axis(gather_axis)\n"
        "    candidate_axes: List[int] = []\n"
        "    if axis_hint_cl is not None and int(axis_hint_cl) != gather_axis:\n"
        "        candidate_axes.append(int(axis_hint_cl))\n"
        "    candidate_axes.extend(\n"
        "        axis\n"
        "        for axis in range(params.ndim - 1, -1, -1)\n"
        "        if axis != gather_axis and (axis_hint_cl is None or axis != int(axis_hint_cl))\n"
        "    )\n"
        "    for candidate_axis in candidate_axes:\n"
        "        candidate_y, candidate_valid = _gather_for_axis(int(candidate_axis))\n"
        "        use_candidate = torch.logical_and(torch.logical_not(y_valid), candidate_valid)\n"
        "        y = torch.where(use_candidate, candidate_y, y)\n"
        "        y_valid = torch.logical_or(y_valid, candidate_valid)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_gather_nd(params: torch.Tensor, indices: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    index_depth = int(indices_i64.shape[-1])\n"
        "    flat_indices = indices_i64.reshape(-1, index_depth)\n"
        "    prefix_shape = list(indices_i64.shape[:-1])\n"
        "    slice_shape = list(params.shape[index_depth:])\n"
        "    if index_depth == 0:\n"
        "        gathered_rows = params\n"
        "    else:\n"
        "        gather_indices = torch.jit.annotate(List[torch.Tensor], [])\n"
        "        for axis in range(index_depth):\n"
        "            gather_indices.append(flat_indices[:, axis])\n"
        "        gathered_rows = params[tuple(gather_indices)]\n"
        "    output_shape = prefix_shape + slice_shape\n"
        "    if len(output_shape) == 0:\n"
        "        y = torch.reshape(gathered_rows, [])\n"
        "    else:\n"
        "        y = torch.reshape(gathered_rows, output_shape)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape: Any, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    output_shape = _shape_list(shape)\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    index_depth = int(indices_i64.shape[-1])\n"
        "    prefix_shape = list(indices_i64.shape[:-1])\n"
        "    slice_shape = list(output_shape[index_depth:])\n"
        "    expected_updates_shape = prefix_shape + slice_shape\n"
        "    aligned_updates = _align_scatter_nd_updates(updates, expected_updates_shape)\n"
        "    flat_indices = indices_i64.reshape(-1, index_depth)\n"
        "    row_shape = [int(v) for v in output_shape[:index_depth]]\n"
        "    strides: List[int] = [1] * len(row_shape)\n"
        "    row_count = 1\n"
        "    running = 1\n"
        "    for axis in range(len(row_shape) - 1, -1, -1):\n"
        "        strides[axis] = int(running)\n"
        "        running *= int(row_shape[axis])\n"
        "    row_count = int(running)\n"
        "    slice_size = 1\n"
        "    for dim in slice_shape:\n"
        "        slice_size *= int(dim)\n"
        "    stride_tensor = torch.as_tensor(strides, dtype=torch.int64, device=indices_i64.device)\n"
        "    flat_rows = torch.sum(flat_indices * stride_tensor, dim=-1)\n"
        "    if int(slice_size) == 1:\n"
        "        y_flat = torch.zeros((row_count,), dtype=aligned_updates.dtype, device=aligned_updates.device)\n"
        "        y_flat.index_copy_(0, flat_rows, aligned_updates.reshape(-1))\n"
        "        y = y_flat.reshape(output_shape)\n"
        "    else:\n"
        "        y_rows = torch.zeros((row_count, slice_size), dtype=aligned_updates.dtype, device=aligned_updates.device)\n"
        "        y_rows.index_copy_(0, flat_rows, aligned_updates.reshape(-1, slice_size))\n"
        "        y = y_rows.reshape(output_shape)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "@torch.jit.ignore\n"
        "def _apply_if_axis0_tensor_mux_export(cond: torch.Tensor, then_value: torch.Tensor, else_value: torch.Tensor) -> torch.Tensor:\n"
        "    predicate = torch.reshape(cond.to(dtype=torch.bool), [])\n"
        "    def _then(branch_then: torch.Tensor, branch_else: torch.Tensor) -> torch.Tensor:\n"
        "        return branch_then.clone()\n"
        "    def _else(branch_then: torch.Tensor, branch_else: torch.Tensor) -> torch.Tensor:\n"
        "        return branch_else.clone()\n"
        "    return torch.cond(predicate, _then, _else, (then_value, else_value))\n\n"
        "def _apply_if_axis0_tensor_mux(cond: torch.Tensor, then_value: torch.Tensor, else_value: torch.Tensor, merged: torch.Tensor, begin: torch.Tensor, size: torch.Tensor, target_shape: Optional[Sequence[int]], use_export_mode: bool) -> torch.Tensor:\n"
        "    if bool(use_export_mode):\n"
        "        y = _apply_if_axis0_tensor_mux_export(cond, then_value, else_value)\n"
        "        return _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_slice(merged, begin, size, target_shape)\n\n"
        "def _apply_slice(x: torch.Tensor, begin: torch.Tensor, size: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    begin_values = _shape_list(begin)\n"
        "    size_values = _shape_list(size)\n"
        "    y = x\n"
        "    for axis, start in enumerate(begin_values):\n"
        "        dim_size = y.shape[axis]\n"
        "        start_index = int(start)\n"
        "        if start_index < 0:\n"
        "            start_index += dim_size\n"
        "        length = int(size_values[axis])\n"
        "        if length < 0:\n"
        "            narrowed_length = dim_size - start_index\n"
        "        else:\n"
        "            narrowed_length = length\n"
        "        y = torch.narrow(y, axis, start_index, narrowed_length)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_strided_slice(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor, strides: torch.Tensor, begin_mask: int, end_mask: int, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    begin_values = _shape_list(begin)\n"
        "    end_values = _shape_list(end)\n"
        "    stride_values = _shape_list(strides)\n"
        "    y = x\n"
        "    for axis, (start, stop, step) in enumerate(zip(begin_values, end_values, stride_values)):\n"
        "        dim_size = y.shape[axis]\n"
        "        has_dynamic_start = (not ((int(begin_mask) >> axis) & 1)) and isinstance(start, torch.Tensor)\n"
        "        has_dynamic_stop = (not ((int(end_mask) >> axis) & 1)) and isinstance(stop, torch.Tensor)\n"
        "        has_dynamic_step = isinstance(step, torch.Tensor)\n"
        "        if (not torch.jit.is_scripting()) and (has_dynamic_start or has_dynamic_stop or has_dynamic_step):\n"
        "            dim_size_tensor = torch.scalar_tensor(dim_size, dtype=torch.int64, device=y.device)\n"
        "            axis_indices = torch.arange(dim_size, dtype=torch.int64, device=y.device)\n"
        "            step_tensor = torch.reshape(torch.as_tensor(step, dtype=torch.int64, device=y.device), [])\n"
        "            if ((int(begin_mask) >> axis) & 1):\n"
        "                start_tensor = torch.zeros([], dtype=torch.int64, device=y.device)\n"
        "            else:\n"
        "                start_tensor = torch.reshape(torch.as_tensor(start, dtype=torch.int64, device=y.device), [])\n"
        "                start_tensor = torch.where(start_tensor < 0, start_tensor + dim_size_tensor, start_tensor)\n"
        "            if ((int(end_mask) >> axis) & 1):\n"
        "                stop_tensor = dim_size_tensor\n"
        "            else:\n"
        "                stop_tensor = torch.reshape(torch.as_tensor(stop, dtype=torch.int64, device=y.device), [])\n"
        "                stop_tensor = torch.where(stop_tensor < 0, stop_tensor + dim_size_tensor, stop_tensor)\n"
        "            axis_mask = torch.logical_and(axis_indices >= start_tensor, axis_indices < stop_tensor)\n"
        "            axis_mask = torch.logical_and(axis_mask, torch.eq(torch.remainder(axis_indices - start_tensor, step_tensor), 0))\n"
        "            axis_indices = axis_indices[axis_mask]\n"
        "            y = torch.index_select(y, axis, axis_indices)\n"
        "            continue\n"
        "        step_value = int(step)\n"
        "        if step_value == 0:\n"
        "            raise RuntimeError('STRIDED_SLICE step must be non-zero')\n"
        "        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)\n"
        "        resolved_stop = None if ((int(end_mask) >> axis) & 1) else int(stop)\n"
        "        if step_value > 0:\n"
        "            start_index = 0 if resolved_start is None else int(resolved_start)\n"
        "            stop_index = dim_size if resolved_stop is None else int(resolved_stop)\n"
        "            if start_index < 0:\n"
        "                start_index += dim_size\n"
        "            if stop_index < 0:\n"
        "                stop_index += dim_size\n"
        "        else:\n"
        "            start_index = dim_size - 1 if resolved_start is None else int(resolved_start)\n"
        "            stop_index = -1 if resolved_stop is None else int(resolved_stop)\n"
        "            if start_index < 0:\n"
        "                start_index += dim_size\n"
        "            if resolved_stop is not None and stop_index < 0:\n"
        "                stop_index += dim_size\n"
        "        if step_value == 1:\n"
        "            narrowed_length = stop_index - start_index\n"
        "            y = torch.narrow(y, axis, start_index, narrowed_length)\n"
        "            continue\n"
        "        axis_indices = torch.arange(start_index, stop_index, step_value, dtype=torch.int64, device=y.device)\n"
        "        y = torch.index_select(y, axis, axis_indices)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _resolve_same_padding(input_size: int, kernel_size: int, stride: int, target_size: Optional[int] = None) -> Tuple[int, int]:\n"
        "    if target_size is None:\n"
        "        out_size = (int(input_size) + int(stride) - 1) // int(stride)\n"
        "    else:\n"
        "        out_size = int(target_size)\n"
        "    total = max((int(out_size) - 1) * int(stride) + int(kernel_size) - int(input_size), 0)\n"
        "    before = total // 2\n"
        "    after = total - before\n"
        "    return before, after\n\n"
        "def _apply_pool2d(x: torch.Tensor, filter_height: int, filter_width: int, stride_h: int, stride_w: int, padding: str, target_shape: Optional[Sequence[int]], is_max_pool: bool, channel_last: Optional[bool] = None) -> torch.Tensor:\n"
        "    resize_as_channel_last = bool(channel_last) if channel_last is not None else False\n"
        "    if channel_last is None:\n"
        "        has_actual_shape, actual_shape = _tensor_static_shape_list(x)\n"
        "        if has_actual_shape:\n"
        "            resize_as_channel_last = bool(\n"
        "                x.ndim == 4\n"
        "                and actual_shape[1] > actual_shape[-1]\n"
        "                and actual_shape[2] > actual_shape[-1]\n"
        "            )\n"
        "    has_target_shape, target = _optional_static_shape_list(target_shape)\n"
        "    if channel_last is None and x.ndim == 4 and has_target_shape and len(target) == 4:\n"
        "        has_actual_shape, actual_shape = _tensor_static_shape_list(x)\n"
        "        if has_actual_shape and (\n"
        "            (actual_shape[-1] == target[1] and actual_shape[1] != target[1])\n"
        "            or (actual_shape[-1] == target[-1] and actual_shape[1] != target[-1])\n"
        "        ):\n"
        "            resize_as_channel_last = True\n"
        "    pool_input = x.permute(0, 3, 1, 2).contiguous() if resize_as_channel_last and x.ndim == 4 else x\n"
        "    if str(padding).upper() == 'SAME':\n"
        "        target_h: Optional[int] = None\n"
        "        target_w: Optional[int] = None\n"
        "        if has_target_shape and len(target) == 4:\n"
        "            if resize_as_channel_last:\n"
        "                if int(target[1]) == int(pool_input.shape[1]) and int(target[-1]) != int(pool_input.shape[1]):\n"
        "                    target_h = int(target[2])\n"
        "                    target_w = int(target[3])\n"
        "                else:\n"
        "                    target_h = int(target[1])\n"
        "                    target_w = int(target[2])\n"
        "            else:\n"
        "                target_h = int(target[2])\n"
        "                target_w = int(target[3])\n"
        "        pad_w = _resolve_same_padding(int(pool_input.shape[-1]), filter_width, stride_w, target_w)\n"
        "        pad_h = _resolve_same_padding(int(pool_input.shape[-2]), filter_height, stride_h, target_h)\n"
        "        pool_input = F.pad(pool_input, [pad_w[0], pad_w[1], pad_h[0], pad_h[1]], mode='constant', value=float('-inf') if is_max_pool else 0.0)\n"
        "        padding_value = 0\n"
        "    else:\n"
        "        padding_value = 0\n"
        "    if is_max_pool:\n"
        "        y = F.max_pool2d(pool_input, kernel_size=(filter_height, filter_width), stride=(stride_h, stride_w), padding=padding_value)\n"
        "    else:\n"
        "        y = F.avg_pool2d(pool_input, kernel_size=(filter_height, filter_width), stride=(stride_h, stride_w), padding=padding_value)\n"
        "    if resize_as_channel_last and y.ndim == 4:\n"
        "        y = y.permute(0, 2, 3, 1).contiguous()\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "@torch.jit.ignore\n"
        "def _onnx_export_active() -> bool:\n"
        "    return bool(torch.onnx.is_in_onnx_export())\n\n"
        "def _resize_bilinear_exact(x: torch.Tensor, size: Sequence[int], *, align_corners: bool, half_pixel_centers: bool) -> torch.Tensor:\n"
        "    if not torch.jit.is_scripting():\n"
        "        if _onnx_export_active():\n"
        "            return F.interpolate(x, size=[int(size[0]), int(size[1])], mode='bilinear', align_corners=align_corners)\n"
        "    if x.ndim != 4:\n"
        "        return F.interpolate(x, size=[int(size[0]), int(size[1])], mode='bilinear', align_corners=align_corners)\n"
        "    out_h = int(size[0])\n"
        "    out_w = int(size[1])\n"
        "    in_h = int(x.shape[-2])\n"
        "    in_w = int(x.shape[-1])\n"
        "    if out_h <= 0 or out_w <= 0:\n"
        "        raise RuntimeError('Resize target dimensions must be positive.')\n"
        "    if align_corners:\n"
        "        ys = torch.zeros([out_h], dtype=torch.float32, device=x.device) if out_h == 1 else torch.arange(out_h, dtype=torch.float32, device=x.device) * float(max(in_h - 1, 0)) / float(max(out_h - 1, 1))\n"
        "        xs = torch.zeros([out_w], dtype=torch.float32, device=x.device) if out_w == 1 else torch.arange(out_w, dtype=torch.float32, device=x.device) * float(max(in_w - 1, 0)) / float(max(out_w - 1, 1))\n"
        "    elif half_pixel_centers:\n"
        "        ys = (torch.arange(out_h, dtype=torch.float32, device=x.device) + 0.5) * float(in_h) / float(out_h) - 0.5\n"
        "        xs = (torch.arange(out_w, dtype=torch.float32, device=x.device) + 0.5) * float(in_w) / float(out_w) - 0.5\n"
        "    else:\n"
        "        ys = torch.arange(out_h, dtype=torch.float32, device=x.device) * float(in_h) / float(out_h)\n"
        "        xs = torch.arange(out_w, dtype=torch.float32, device=x.device) * float(in_w) / float(out_w)\n"
        "    y0 = torch.floor(ys).to(dtype=torch.int64)\n"
        "    x0 = torch.floor(xs).to(dtype=torch.int64)\n"
        "    y1 = y0 + 1\n"
        "    x1 = x0 + 1\n"
        "    y0c = y0.clamp(0, max(in_h - 1, 0))\n"
        "    x0c = x0.clamp(0, max(in_w - 1, 0))\n"
        "    y1c = y1.clamp(0, max(in_h - 1, 0))\n"
        "    x1c = x1.clamp(0, max(in_w - 1, 0))\n"
        "    ly = (ys - y0.to(dtype=torch.float32)).view(1, 1, out_h, 1)\n"
        "    lx = (xs - x0.to(dtype=torch.float32)).view(1, 1, 1, out_w)\n"
        "    hy = 1.0 - ly\n"
        "    hx = 1.0 - lx\n"
        "    top_left = x[:, :, y0c[:, None], x0c[None, :]]\n"
        "    top_right = x[:, :, y0c[:, None], x1c[None, :]]\n"
        "    bottom_left = x[:, :, y1c[:, None], x0c[None, :]]\n"
        "    bottom_right = x[:, :, y1c[:, None], x1c[None, :]]\n"
        "    return top_left * hy * hx + top_right * hy * lx + bottom_left * ly * hx + bottom_right * ly * lx\n\n"
        "def _apply_resize(x: torch.Tensor, size: Any, method: str, target_shape: Optional[Sequence[int]], align_corners: bool = False, half_pixel_centers: bool = False, channel_last: Optional[bool] = None) -> torch.Tensor:\n"
        "    resize_size = _shape_list(size)\n"
        "    resize_as_channel_last = bool(channel_last) if channel_last is not None else False\n"
        "    has_target_shape, target = _optional_static_shape_list(target_shape)\n"
        "    if channel_last is None and x.ndim == 4 and has_target_shape and len(target) == 4:\n"
        "        has_actual_shape, actual_shape = _tensor_static_shape_list(x)\n"
        "        if has_actual_shape and (\n"
        "            (actual_shape[-1] == target[1] and actual_shape[1] != target[1])\n"
        "            or (actual_shape[-1] == target[-1] and actual_shape[1] != target[-1])\n"
        "        ):\n"
        "            resize_as_channel_last = True\n"
        "    resize_input = x.permute(0, 3, 1, 2).contiguous() if resize_as_channel_last and x.ndim == 4 else x\n"
        "    if str(method).lower() == 'nearest':\n"
        "        y = F.interpolate(resize_input, size=resize_size, mode='nearest')\n"
        "    else:\n"
        "        y = _resize_bilinear_exact(resize_input, resize_size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)\n"
        "    if resize_as_channel_last and y.ndim == 4:\n"
        "        y = y.permute(0, 2, 3, 1).contiguous()\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _box_iou(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:\n"
        "    x1 = torch.maximum(boxes[:, 0], box[0])\n"
        "    y1 = torch.maximum(boxes[:, 1], box[1])\n"
        "    x2 = torch.minimum(boxes[:, 2], box[2])\n"
        "    y2 = torch.minimum(boxes[:, 3], box[3])\n"
        "    inter_w = torch.clamp(x2 - x1, min=0.0)\n"
        "    inter_h = torch.clamp(y2 - y1, min=0.0)\n"
        "    inter = inter_w * inter_h\n"
        "    boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)\n"
        "    box_area = torch.clamp(box[2] - box[0], min=0.0) * torch.clamp(box[3] - box[1], min=0.0)\n"
        "    union = boxes_area + box_area - inter\n"
        "    safe_union = torch.where(union > 0, union, torch.ones_like(union))\n"
        "    iou = inter / safe_union\n"
        "    return torch.where(union > 0, iou, torch.zeros_like(iou))\n\n"
        "def _apply_non_max_suppression_v4(boxes: torch.Tensor, scores: torch.Tensor, max_output_size: torch.Tensor, iou_threshold: torch.Tensor, score_threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    flat_boxes = boxes.to(dtype=torch.float32).reshape(-1, 4)\n"
        "    flat_scores = scores.to(dtype=torch.float32).reshape(-1)\n"
        "    max_outputs = max(0, int(max_output_size.reshape(-1)[0].to(dtype=torch.int64).item()))\n"
        "    selected = torch.zeros([max_outputs], dtype=torch.int64, device=flat_boxes.device)\n"
        "    valid_count = torch.zeros([], dtype=torch.int32, device=flat_boxes.device)\n"
        "    if max_outputs == 0:\n"
        "        return selected.to(dtype=torch.int32), valid_count\n"
        "    iou_thresh = float(iou_threshold.reshape(-1)[0].item())\n"
        "    score_thresh = float(score_threshold.reshape(-1)[0].item())\n"
        "    candidate_scores = torch.where(\n"
        "        flat_scores > score_thresh,\n"
        "        flat_scores,\n"
        "        torch.full_like(flat_scores, float('-inf')),\n"
        "    )\n"
        "    all_indices = torch.arange(flat_scores.shape[0], dtype=torch.int64, device=flat_boxes.device)\n"
        "    neg_inf = torch.full_like(candidate_scores, float('-inf'))\n"
        "    for out_index in range(max_outputs):\n"
        "        current_score, current_index = torch.max(candidate_scores, dim=0)\n"
        "        current_index = current_index.to(dtype=torch.int64)\n"
        "        is_valid = torch.isfinite(current_score)\n"
        "        selected[out_index : out_index + 1] = torch.where(is_valid, current_index, torch.zeros_like(current_index)).reshape(1)\n"
        "        valid_count = valid_count + is_valid.to(dtype=torch.int32)\n"
        "        current_box = torch.index_select(flat_boxes, 0, current_index.reshape(1)).reshape(4)\n"
        "        suppress = _box_iou(flat_boxes, current_box) > iou_thresh\n"
        "        suppress = torch.logical_or(suppress, all_indices == current_index)\n"
        "        suppress = torch.logical_and(suppress, is_valid)\n"
        "        candidate_scores = torch.where(suppress, neg_inf, candidate_scores)\n"
        "    return selected.to(dtype=torch.int32), valid_count\n\n"
        "def _crop_nms_selected_indices(selected_indices: torch.Tensor, valid_count: torch.Tensor) -> torch.Tensor:\n"
        "    limit = valid_count.to(dtype=torch.int64).reshape([])\n"
        "    return torch.narrow(selected_indices, 0, 0, limit)\n\n"
        "def _normalize_axes(value: Any, rank: int) -> Optional[List[int]]:\n"
        "    if value is None:\n"
        "        return None\n"
        "    axes = _shape_list(value)\n"
        "    normalized = torch.jit.annotate(List[int], [])\n"
        "    for raw_axis in axes:\n"
        "        axis = _normalize_dim(int(raw_axis), rank)\n"
        "        insert_at = 0\n"
        "        while insert_at < len(normalized) and int(normalized[insert_at]) < axis:\n"
        "            insert_at += 1\n"
        "        if insert_at < len(normalized) and int(normalized[insert_at]) == axis:\n"
        "            continue\n"
        "        normalized.insert(insert_at, axis)\n"
        "    return normalized\n\n"
        "def _reduce_sum(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.sum(x) if not keepdims else torch.sum(x).reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    if keepdims:\n"
        "        for dim in axis:\n"
        "            result = torch.sum(result, dim=int(dim), keepdim=True)\n"
        "        return result\n"
        "    reverse_index = len(axis) - 1\n"
        "    while reverse_index >= 0:\n"
        "        result = torch.sum(result, dim=int(axis[reverse_index]), keepdim=False)\n"
        "        reverse_index -= 1\n"
        "    return result\n\n"
        "def _reduce_mean(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.mean(x) if not keepdims else torch.mean(x).reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    if keepdims:\n"
        "        for dim in axis:\n"
        "            result = torch.mean(result, dim=int(dim), keepdim=True)\n"
        "        return result\n"
        "    reverse_index = len(axis) - 1\n"
        "    while reverse_index >= 0:\n"
        "        result = torch.mean(result, dim=int(axis[reverse_index]), keepdim=False)\n"
        "        reverse_index -= 1\n"
        "    return result\n\n"
        "def _reduce_max(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.amax(x, keepdim=keepdims)\n"
        "    result = x\n"
        "    if keepdims:\n"
        "        for dim in axis:\n"
        "            result = torch.amax(result, dim=int(dim), keepdim=True)\n"
        "        return result\n"
        "    reverse_index = len(axis) - 1\n"
        "    while reverse_index >= 0:\n"
        "        result = torch.amax(result, dim=int(axis[reverse_index]), keepdim=False)\n"
        "        reverse_index -= 1\n"
        "    return result\n\n"
        "def _reduce_min(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.amin(x, keepdim=keepdims)\n"
        "    result = x\n"
        "    if keepdims:\n"
        "        for dim in axis:\n"
        "            result = torch.amin(result, dim=int(dim), keepdim=True)\n"
        "        return result\n"
        "    reverse_index = len(axis) - 1\n"
        "    while reverse_index >= 0:\n"
        "        result = torch.amin(result, dim=int(axis[reverse_index]), keepdim=False)\n"
        "        reverse_index -= 1\n"
        "    return result\n\n"
        "def _reduce_prod(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        y = torch.prod(x)\n"
        "        return y if not keepdims else y.reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    if keepdims:\n"
        "        for dim in axis:\n"
        "            result = torch.prod(result, dim=int(dim), keepdim=True)\n"
        "        return result\n"
        "    reverse_index = len(axis) - 1\n"
        "    while reverse_index >= 0:\n"
        "        result = torch.prod(result, dim=int(axis[reverse_index]), keepdim=False)\n"
        "        reverse_index -= 1\n"
        "    return result\n\n"
        "def _reduce_any(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        y = torch.any(x)\n"
        "        return y if not keepdims else y.reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    if keepdims:\n"
        "        for dim in axis:\n"
        "            result = torch.any(result, dim=int(dim), keepdim=True)\n"
        "        return result\n"
        "    reverse_index = len(axis) - 1\n"
        "    while reverse_index >= 0:\n"
        "        result = torch.any(result, dim=int(axis[reverse_index]), keepdim=False)\n"
        "        reverse_index -= 1\n"
        "    return result\n\n"
    ) + conv_block_helper_source

    runtime_source = _build_native_runtime_source(helper_source)
    _write_generated_package_common_files(
        output_folder_path,
        runtime_source=runtime_source,
    )
    buffer_init_block = "\n".join(f"        {line}" for line in buffer_init_lines)
    buffer_annotation_block = ""
    init_constants_method = ""
    if len(buffer_init_lines) > 0:
        init_constants_method = (
            "    def _init_constants(self) -> None:\n"
            f"{buffer_init_block}\n\n"
        )
    init_constants_call = "        self._init_constants()\n" if len(buffer_init_lines) > 0 else ""
    forward_kwargs_lines: List[str] = []
    forward_args_lines: List[str] = []
    forward_signature_args: List[str] = []
    for input_index, input_name in enumerate(model_ir.inputs):
        input_var = tensor_var_names[str(input_name)]
        forward_signature_args.append(f"{input_var}: torch.Tensor")
        forward_kwargs_lines.append(
            f"            {input_var} = resolve_named_input_value(kwargs, {str(input_name)!r})"
        )
        forward_args_lines.append(
            f"            {input_var} = args[{input_index}]"
        )

    forward_lines = _fold_channel_last_affine_conv_bridges(
        forward_lines,
        derive_local_var_name=_derived_local_var_name,
        channel_first_constant_expr_for_buffer_attr=_channel_first_constant_expr_for_buffer_attr,
        skipped_module_names={
            str(attr_name)
            for op_index, attr_name in op_module_attr_names.items()
            if str(model_ir.operators[int(op_index)].op_type) == "DEPTHWISE_CONV_2D"
        },
    )
    forward_lines = _rewrite_channel_last_binary_bridge_chains(
        forward_lines,
        derive_local_var_name=_derived_local_var_name,
        channel_first_constant_expr_for_buffer_attr=_channel_first_constant_expr_for_buffer_attr,
        skipped_module_names={
            str(attr_name)
            for op_index, attr_name in op_module_attr_names.items()
            if str(model_ir.operators[int(op_index)].op_type) == "DEPTHWISE_CONV_2D"
        },
    )
    forward_lines = _fold_channel_first_gap_conv_bridges(forward_lines)
    forward_lines = _rewrite_channel_first_gap_outputs_to_explicit_channel_last(forward_lines)
    forward_lines = _rewrite_channel_last_gap_means_to_reduce_mean(forward_lines)
    forward_lines = _fold_boundary_transpose_pad_conv_bridges(forward_lines)
    bridged_forward_lines = _bridge_boundary_metadata_gather_nd_inputs(
        forward_lines,
        model_ir=model_ir,
        tensor_var_names=tensor_var_names,
    )
    if bridged_forward_lines != forward_lines:
        runtime_imports.add("_torch_permute")
    forward_lines = bridged_forward_lines
    collapsed_forward_lines = _collapse_redundant_torch_permute_chains(forward_lines)
    if collapsed_forward_lines != forward_lines:
        runtime_imports.add("_torch_permute")
    forward_lines = collapsed_forward_lines
    forward_lines = _fold_channel_last_prelu_bridges(forward_lines)
    forward_lines = _inline_trivial_public_layout_bridge_aliases(forward_lines)
    forward_lines = _fold_rank4_reshape_permute_conv_bridges(forward_lines)
    forward_lines = _ensure_explicit_depthwise_channel_last_input_bridges(
        forward_lines,
        module_names={
            str(attr_name)
            for op_index, attr_name in op_module_attr_names.items()
            if (
                str(model_ir.operators[int(op_index)].op_type) == "DEPTHWISE_CONV_2D"
                and is_channel_last_logical_layout(
                    normalize_logical_layout(
                        model_ir.tensors[str(model_ir.operators[int(op_index)].inputs[0])].logical_layout
                    )
                )
            )
        },
    )
    forward_lines = _fold_single_use_static_reshape_chains(
        forward_lines,
        tensor_var_names=tensor_var_names,
        model_ir=model_ir,
    )
    forward_lines = _prune_dead_forward_lines(
        forward_lines,
        input_var_names=[str(tensor_var_names[str(name)]) for name in model_ir.inputs],
        output_var_names=[str(tensor_var_names[str(name)]) for name in model_ir.outputs],
    )
    stage_methods_source, forward_stage_calls, stage_specs = _build_forward_stage_methods(
        forward_lines,
        tensor_var_names=tensor_var_names,
        model_ir=model_ir,
    )
    named_encoder_class_source = ""
    named_encoder_init_lines: List[str] = []
    if len(stage_specs) > 0:
        named_encoder_class_source, named_encoder_init_lines, forward_stage_calls = _build_named_encoder_methods_composite(
            stage_specs,
            final_output_names={str(tensor_var_names[str(name)]) for name in model_ir.outputs},
        )
    bridged_forward_stage_calls = _bridge_boundary_metadata_gather_nd_inputs(
        [str(line).strip() for line in forward_stage_calls],
        model_ir=model_ir,
        tensor_var_names=tensor_var_names,
    )
    collapsed_forward_stage_calls = _collapse_redundant_torch_permute_chains(
        bridged_forward_stage_calls
    )
    if collapsed_forward_stage_calls != bridged_forward_stage_calls:
        bridged_forward_stage_calls = collapsed_forward_stage_calls
    bridged_forward_stage_calls = _inline_trivial_public_layout_bridge_aliases(
        bridged_forward_stage_calls
    )
    if bridged_forward_stage_calls != [str(line).strip() for line in forward_stage_calls]:
        runtime_imports.add("_torch_permute")
        forward_stage_calls = [f"        {line}" for line in bridged_forward_stage_calls]
    if len(named_encoder_init_lines) > 0:
        module_init_lines.extend(named_encoder_init_lines)
    module_init_block = "\n".join(f"        {line}" for line in module_init_lines)
    forward_block = "\n".join(forward_stage_calls)
    forward_kwargs_block = "\n".join(forward_kwargs_lines) if len(forward_kwargs_lines) > 0 else "            pass"
    forward_args_block = "\n".join(forward_args_lines) if len(forward_args_lines) > 0 else "            pass"
    forward_signature = ", ".join(["self"] + forward_signature_args)
    forward_named_call_args = ", ".join(
        str(tensor_var_names[str(input_name)])
        for input_name in model_ir.inputs
    )
    outputs_expr = ", ".join(
        (
            tensor_var_names[str(name)]
            if str(name) in producer_index or str(name) in model_ir.inputs
            else _tensor_expr(str(name))
        )
        for name in model_ir.outputs
    )
    runtime_import_order = [
        "_Conv2dBlock",
        "_align_binary_inputs",
        "_align_binary_inputs_to_anchor",
        "_align_tensor_to_target_shape",
        "_apply_concat",
        "_apply_cumsum",
        "_apply_if_axis0_tensor_mux",
        "_crop_nms_selected_indices",
        "_apply_fused_activation",
        "_apply_gather",
        "_apply_gather_elements",
        "_apply_gather_nd",
        "_apply_random_standard_normal",
        "_apply_scatter_nd",
        "_apply_tile",
        "_apply_module_conv2d",
        "_apply_module_conv3d",
        "_apply_module_transpose_conv2d",
        "_apply_module_transpose_conv3d",
        "_apply_non_max_suppression_v4",
        "_apply_pool2d",
        "_apply_pad_nd",
        "_apply_resize",
        "_apply_slice",
        "_apply_softmax",
        "_apply_strided_slice",
        "_coerce_scalar_axis",
        "_module_device",
        "_normalize_axes",
        "_normalize_dim",
        "_reduce_any",
        "_reduce_max",
        "_reduce_mean",
        "_reduce_min",
        "_reduce_prod",
        "_reduce_sum",
        "_reshape_gather_output",
        "_resolve_reshape_shape",
        "_resolve_reshape_shape_tensor",
        "resolve_named_input_value",
        "_shape_list",
        "_tensor_shape_list",
        "_shape_tensor",
        "_to_torch_pad_arg",
        "_torch_dtype",
        "_torch_permute",
        "load_generated_weights",
    ]
    if has_conv_blocks:
        runtime_imports.add("_Conv2dBlock")
    has_affine_layer_norms = len(affine_layer_norm_specs) > 0
    affine_layer_norm_source = (
        "class _AffineLayerNorm(torch.nn.Module):\n"
        "    gamma: torch.Tensor\n"
        "    beta: torch.Tensor\n\n"
        "    def __init__(self, *, shape: list[int], dtype: torch.dtype) -> None:\n"
        "        super().__init__()\n"
        "        self.register_buffer('gamma', torch.zeros(shape, dtype=dtype), persistent=True)\n"
        "        self.register_buffer('beta', torch.zeros(shape, dtype=dtype), persistent=True)\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return torch.add(torch.mul(x, self.gamma), self.beta)\n\n"
    ) if has_affine_layer_norms else ""
    nms_method_source = ""
    if len(nms_method_specs) > 0:
        runtime_imports.add("_module_device")
        method_chunks: List[str] = []
        for spec in nms_method_specs:
            extra_args: List[str] = []
            if spec.get("max_output_arg_expr", None) is not None:
                extra_args.append("max_output_size: torch.Tensor")
            if spec.get("iou_threshold_arg_expr", None) is not None:
                extra_args.append("iou_threshold: torch.Tensor")
            if spec.get("score_threshold_arg_expr", None) is not None:
                extra_args.append("score_threshold: torch.Tensor")
            extra_signature = ""
            if len(extra_args) > 0:
                extra_signature = ", " + ", ".join(extra_args)
            method_chunks.append(
                "    def {name}(self, boxes: torch.Tensor, scores: torch.Tensor{extra_signature}) -> tuple[torch.Tensor, torch.Tensor]:\n"
                "        return _apply_non_max_suppression_v4(\n"
                "            boxes,\n"
                "            scores,\n"
                "            torch.as_tensor({max_output_expr}, dtype=torch.int32, device=_module_device(self)),\n"
                "            torch.as_tensor({iou_threshold_expr}, dtype=torch.float32, device=_module_device(self)),\n"
                "            torch.as_tensor({score_threshold_expr}, dtype=torch.float32, device=_module_device(self)),\n"
                "        )\n\n".format(
                    name=str(spec["name"]),
                    extra_signature=extra_signature,
                    max_output_expr=str(spec["max_output_expr"]),
                    iou_threshold_expr=str(spec["iou_threshold_expr"]),
                    score_threshold_expr=str(spec["score_threshold_expr"]),
                )
            )
        nms_method_source = "".join(method_chunks)
    runtime_import_block = "".join(
        f"    {name},\n" for name in runtime_import_order if name in runtime_imports
    )
    constant_buffer_alias_method_source = ""
    constant_buffer_alias_init_call = ""
    constant_buffer_alias_load_state_dict_source = ""
    if (
        len(channel_first_constant_buffer_alias_refresh_specs) > 0
        or len(permuted_constant_buffer_alias_refresh_specs) > 0
        or len(transposed_constant_buffer_alias_refresh_specs) > 0
    ):
        refresh_lines = [
            "    def _refresh_constant_buffer_aliases(self) -> None:\n",
            "        with torch.no_grad():\n",
        ]
        for alias_attr_name, source_attr_name, alias_shape in channel_first_constant_buffer_alias_refresh_specs:
            refresh_lines.append(
                f"            self.{alias_attr_name}.copy_(torch.reshape(self.{source_attr_name}, {repr(alias_shape)}))\n"
            )
        for alias_attr_name, source_attr_name, perm in permuted_constant_buffer_alias_refresh_specs:
            refresh_lines.append(
                f"            self.{alias_attr_name}.copy_(self.{source_attr_name}.permute(*{repr(tuple(int(v) for v in list(perm)))}).contiguous())\n"
            )
        for alias_attr_name, source_attr_name in transposed_constant_buffer_alias_refresh_specs:
            refresh_lines.append(
                f"            self.{alias_attr_name}.copy_(self.{source_attr_name}.transpose(-1, -2))\n"
            )
        refresh_lines.append("\n")
        constant_buffer_alias_method_source = "".join(refresh_lines)
        constant_buffer_alias_init_call = "        self._refresh_constant_buffer_aliases()\n"
        constant_buffer_alias_load_state_dict_source = (
            "    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, assign: bool = False):\n"
            "        result = super().load_state_dict(state_dict, strict=strict, assign=assign)\n"
            "        self._refresh_constant_buffer_aliases()\n"
            "        return result\n\n"
        )

    model_source = (
        "# pyright: reportArgumentType=false, reportCallIssue=false\n"
        "from pathlib import Path\n"
        "from typing import Any, Callable, Dict, Optional, Tuple\n\n"
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        "from .runtime import (\n"
        f"{runtime_import_block}"
        ")\n\n"
        "PACKAGE_DIR = Path(__file__).resolve().parent\n"
        f"INPUT_NAMES = {repr([str(v) for v in model_ir.inputs])}\n"
        f"OUTPUT_NAMES = {repr([str(v) for v in model_ir.outputs])}\n"
        f"{sequence_rnn_helper_source}"
        f"{sequence_lstm_helper_source}"
        f"{affine_layer_norm_source}"
        f"{named_encoder_class_source}"
        "class Model(torch.nn.Module):\n"
        f"{buffer_annotation_block}"
        "    def __init__(self, *, device: str | None = None, eval_mode: bool = True, load_weights: bool = True):\n"
        "        super().__init__()\n"
        "        self.input_names = list(INPUT_NAMES)\n"
        "        self.output_names = list(OUTPUT_NAMES)\n"
        "        self._onnx2tf_torch_export_mode = False\n"
        f"{module_init_block}\n"
        f"{init_constants_call}"
        "        if load_weights:\n"
        "            load_generated_weights(\n"
        "                model=self,\n"
        "                package_dir=PACKAGE_DIR,\n"
        "                device=device,\n"
        "            )\n"
        "        elif device is not None:\n"
        "            self.to(device)\n"
        f"{constant_buffer_alias_init_call}"
        "        if eval_mode:\n"
        "            self.eval()\n\n"
        f"{init_constants_method}"
        f"{constant_buffer_alias_method_source}"
        f"{constant_buffer_alias_load_state_dict_source}"
        f"{nms_method_source}"
        f"{stage_methods_source}"
        f"    def forward({forward_signature}) -> Any:\n"
        f"{forward_block}\n"
    )
    if len(model_ir.outputs) == 1:
        model_source += (
            f"        return {outputs_expr}\n\n"
            "    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:\n"
            "        if len(args) > 0 and len(kwargs) > 0:\n"
            "            raise RuntimeError('Use either positional inputs or keyword inputs, not both.')\n"
            "        if len(kwargs) > 0:\n"
            f"{forward_kwargs_block}\n"
            "        else:\n"
            f"            if len(args) != {len(model_ir.inputs)}:\n"
            "                raise RuntimeError(f'Input arity mismatch. expected={len(self.input_names)} actual={len(args)}')\n"
            f"{forward_args_block}\n"
            f"        return {{{str(model_ir.outputs[0])!r}: self.forward({forward_named_call_args})}}\n\n"
        )
    else:
        model_source += (
            f"        return ({outputs_expr})\n\n"
            "    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:\n"
            "        if len(args) > 0 and len(kwargs) > 0:\n"
            "            raise RuntimeError('Use either positional inputs or keyword inputs, not both.')\n"
            "        if len(kwargs) > 0:\n"
            f"{forward_kwargs_block}\n"
            "        else:\n"
            f"            if len(args) != {len(model_ir.inputs)}:\n"
            "                raise RuntimeError(f'Input arity mismatch. expected={len(self.input_names)} actual={len(args)}')\n"
            f"{forward_args_block}\n"
            f"        result = self.forward({forward_named_call_args})\n"
            f"        return {{name: value for name, value in zip({repr([str(v) for v in model_ir.outputs])}, result)}}\n\n"
        )
    model_source += (
        "def load_model(device: str | None = None, eval_mode: bool = True) -> Model:\n"
        "    return Model(device=device, eval_mode=eval_mode)\n"
    )
    (package_dir / "model.py").write_text(model_source, encoding="utf-8")
    return [(str(attr_path), str(tensor_name)) for attr_path, tensor_name in load_specs]
