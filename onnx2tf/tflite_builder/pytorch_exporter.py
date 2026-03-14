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
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
import onnx

from onnx2tf.tflite_builder._pytorch_exporter_native_codegen_common import (
    _NativeCodegenBindings,
    _NativeCodegenState,
    _NativeModelFileWriterContext,
)
from onnx2tf.tflite_builder._pytorch_exporter_native_codegen_pipeline import (
    execute_native_codegen_pipeline,
)
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


class ModelIRPyTorchExportError(RuntimeError):
    pass


def _prepare_native_codegen_state(
    context: _NativeModelFileWriterContext,
) -> _NativeCodegenState:
    state = _NativeCodegenState(context=context)
    state.used_local_var_names = set(context.tensor_var_names.values())
    state.public_input_names = {str(name) for name in list(context.model_ir.inputs)}
    state.public_layout_bridge_tensor_names = {
        str(name)
        for name in list(context.model_ir.metadata.get("public_layout_bridge_tensor_names", []))
        if str(name) != ""
    }
    return state


def _build_native_codegen_bindings(
    state: _NativeCodegenState,
) -> _NativeCodegenBindings:
    _ = state
    return _NativeCodegenBindings(module_globals=dict(globals()))


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
    execute_native_codegen_pipeline(state, bindings)


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


def _shape_lists_equal(lhs: Optional[Sequence[int]], rhs: Optional[Sequence[int]]) -> bool:
    if lhs is None or rhs is None:
        return False
    return [int(v) for v in list(lhs)] == [int(v) for v in list(rhs)]


def _shape_lists_equal_relaxed(lhs: Optional[Sequence[int]], rhs: Optional[Sequence[int]]) -> bool:
    if lhs is None or rhs is None:
        return False
    lhs_items = [int(v) for v in list(lhs)]
    rhs_items = [int(v) for v in list(rhs)]
    if len(lhs_items) != len(rhs_items):
        return False
    for lhs_dim, rhs_dim in zip(lhs_items, rhs_items):
        if lhs_dim == rhs_dim:
            continue
        if lhs_dim <= 0 or rhs_dim <= 0:
            continue
        return False
    return True


def _shape_can_broadcast_to_target_relaxed(
    shape: Optional[Sequence[int]],
    target_shape: Optional[Sequence[int]],
) -> bool:
    if shape is None or target_shape is None:
        return False
    shape_items = [int(v) for v in list(shape)]
    target_items = [int(v) for v in list(target_shape)]
    if len(shape_items) != len(target_items):
        return False
    for shape_dim, target_dim in zip(shape_items, target_items):
        if shape_dim == 1 or shape_dim == target_dim:
            continue
        if shape_dim <= 0 or target_dim <= 0:
            continue
        return False
    return True


def _broadcast_shapes_relaxed(
    lhs: Optional[Sequence[int]],
    rhs: Optional[Sequence[int]],
) -> Optional[List[int]]:
    if lhs is None or rhs is None:
        return None
    lhs_items = [int(v) for v in list(lhs)]
    rhs_items = [int(v) for v in list(rhs)]
    if len(lhs_items) != len(rhs_items):
        return None
    result: List[int] = []
    for lhs_dim, rhs_dim in zip(lhs_items, rhs_items):
        if lhs_dim == rhs_dim:
            result.append(int(lhs_dim))
            continue
        if lhs_dim == 1:
            result.append(int(rhs_dim))
            continue
        if rhs_dim == 1:
            result.append(int(lhs_dim))
            continue
        if lhs_dim <= 0 and rhs_dim > 0:
            result.append(int(rhs_dim))
            continue
        if rhs_dim <= 0 and lhs_dim > 0:
            result.append(int(lhs_dim))
            continue
        if lhs_dim <= 0 and rhs_dim <= 0:
            result.append(-1)
            continue
        return None
    return result


def _product_expr(items: Sequence[str]) -> str:
    item_list = [str(item) for item in list(items)]
    if len(item_list) == 0:
        return "1"
    expr = item_list[0]
    for item in item_list[1:]:
        expr = f"({expr} * {item})"
    return expr


def _is_all_ones_shape(shape: Sequence[int]) -> bool:
    values = [int(v) for v in list(shape)]
    return len(values) > 0 and all(int(v) == 1 for v in values)


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


def _tensor_name_suggests_channel_last_layout_for_codegen(tensor_name: str) -> bool:
    return str(tensor_name).lower().endswith(("_nhwc", "_nwc", "_ndhwc"))


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
    tensor_layout = (
        normalize_logical_layout(tensor.logical_layout)
        if tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    if (
        str(tensor_name) in channel_first_tensor_expr_aliases
        and (
            is_channel_last_logical_layout(tensor_layout)
            or (
                tensor_layout == LOGICAL_LAYOUT_UNKNOWN
                and _tensor_name_suggests_channel_last_layout_for_codegen(str(tensor_name))
            )
        )
    ):
        return [int(tensor_shape[0]), int(tensor_shape[3]), int(tensor_shape[1]), int(tensor_shape[2])]
    if is_channel_last_logical_layout(tensor_layout):
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
    tensor_layout = (
        normalize_logical_layout(tensor.logical_layout)
        if tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    if (
        str(tensor_name) in channel_first_tensor_expr_aliases
        and (
            is_channel_last_logical_layout(tensor_layout)
            or (
                tensor_layout == LOGICAL_LAYOUT_UNKNOWN
                and _tensor_name_suggests_channel_last_layout_for_codegen(str(tensor_name))
            )
        )
        and perm_to_cf is not None
    ):
        permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
        if permuted_shape is not None:
            return [int(v) for v in list(permuted_shape)]
    if (
        is_channel_last_logical_layout(tensor_layout)
        and perm_to_cf is not None
    ):
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
    non_singleton_axes = [idx for idx, dim in enumerate(tensor_shape) if int(dim) > 1]
    try:
        np.broadcast_shapes(tuple(tensor_broadcast_shape), tuple(other_broadcast_shape))
        # Keep singleton-expanded constants on their original axis when they
        # already broadcast correctly. Permuting them to exactly match the peer
        # tensor can collapse the intended broadcast result, e.g. [1,384,1]
        # with [1,1,384] should stay as-is and broadcast to [1,384,384].
        if len(non_singleton_axes) == 1:
            return None
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
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    if is_channel_first_logical_layout(tensor_layout):
        return tensor_expr_fn(str(tensor_name))
    alias_expr = channel_first_tensor_expr_aliases.get(str(tensor_name), None)
    if alias_expr is not None:
        return str(alias_expr)
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
    if str(lhs_name) in runtime_shape_uncertain_tensors and _is_all_ones_shape(rhs_shape):
        return "lhs"
    if str(rhs_name) in runtime_shape_uncertain_tensors and _is_all_ones_shape(lhs_shape):
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
            if other_tensor is not None:
                other_layout = normalize_logical_layout(other_tensor.logical_layout)
                other_perm_to_cf = None
                if is_channel_last_logical_layout(other_layout):
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
            perm_to_cf = _perm_cl_to_cf(rank) if is_channel_last_logical_layout(tensor_layout) else None
            if perm_to_cf is not None:
                permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
                base_expr = tensor_expr_fn(str(tensor_name))
                buffer_alias_expr = channel_first_constant_buffer_alias_exprs.get(str(tensor_name), None)
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
    alias_expr = binary_constant_buffer_alias_exprs.get((str(tensor_name), str(other_tensor_name)), None)
    expr = tensor_expr_fn(tensor_name)
    tensor = model_ir.tensors.get(str(tensor_name), None)
    other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
    if tensor is None or other_tensor is None:
        if alias_expr is not None:
            return str(alias_expr)
        return expr
    if alias_expr is not None:
        channel_first_alias_expr = channel_first_constant_buffer_alias_exprs.get(str(tensor_name), None)
        channel_first_alias_shape = channel_first_rank4_constant_buffer_alias_shape_fn(str(tensor_name))
        other_layout = normalize_logical_layout(other_tensor.logical_layout)
        other_prefers_channel_first = (
            str(other_tensor_name) in channel_first_tensor_expr_aliases
            or is_channel_first_logical_layout(other_layout)
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
    if op_type in {"MAXIMUM", "MINIMUM"}:
        rhs_scalar_literal = None
    rhs_expr = rhs_scalar_literal or binary_operand_expr_fn(rhs_name, lhs_name)
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

    if can_emit_channel_first_binary_op_fn(op) and runtime_shape_passthrough_operand is None and not requires_runtime_alignment:
        output_name = str(outputs[0])
        output_tensor = model_ir.tensors.get(output_name, None)
        output_var = output_vars[0]
        output_rank = len(list(output_tensor.shape)) if output_tensor is not None else 0
        output_layout = normalize_logical_layout(
            output_tensor.logical_layout if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        )
        raw_output_var = (
            output_var
            if output_layout == LOGICAL_LAYOUT_UNKNOWN or output_name in set(model_ir.outputs)
            else derived_local_var_name_fn(f"{output_var}_cf", "t")
        )
        lhs_cf_expr = channel_first_binary_input_expr_fn(lhs_name, rhs_name)
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
) -> List[str]:
    if len(lines) < 6:
        return [str(line) for line in lines]

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
) -> List[str]:
    rewritten: List[str] = []
    line_count = len(lines)
    index = 0
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


def _is_layout_only_transpose_by_shape(
    *,
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
    perm: Optional[Sequence[int]],
) -> bool:
    if input_tensor is None or output_tensor is None or perm is None:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if len(input_shape) != len(output_shape) or len(input_shape) != len(list(perm)):
        return False
    return _permute_shape(input_shape, perm) == output_shape


def _is_standard_channel_layout_permutation(
    *,
    perm: Optional[Sequence[int]],
    rank: int,
) -> bool:
    if perm is None:
        return False
    perm_values = tuple(int(v) for v in list(perm))
    return perm_values in {
        tuple(_perm_cl_to_cf(rank) or []),
        tuple(_perm_cf_to_cl(rank) or []),
    }


def _is_inconsistent_standard_layout_transpose(
    *,
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
    perm: Optional[Sequence[int]],
) -> bool:
    if input_tensor is None or output_tensor is None or perm is None:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    rank = len(input_shape)
    if rank not in {3, 4, 5} or len(output_shape) != rank:
        return False
    if not _is_standard_channel_layout_permutation(perm=perm, rank=rank):
        return False
    if input_shape != output_shape:
        return False
    permuted_input_shape = _permute_shape(input_shape, perm)
    if permuted_input_shape is None:
        return False
    # Some layout-bridge transposes survive normalization with stale CF metadata.
    # Executing those transposes would violate the declared tensor shape contract.
    if permuted_input_shape != output_shape:
        return True
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if input_layout == LOGICAL_LAYOUT_UNKNOWN or output_layout == LOGICAL_LAYOUT_UNKNOWN:
        return False
    if input_layout != output_layout:
        return False
    return False


def _is_inconsistent_same_layout_transpose(
    *,
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
    perm: Optional[Sequence[int]],
) -> bool:
    if input_tensor is None or output_tensor is None or perm is None:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    rank = len(input_shape)
    if rank not in {3, 4, 5} or len(output_shape) != rank:
        return False
    if input_shape != output_shape:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if input_layout == LOGICAL_LAYOUT_UNKNOWN or output_layout == LOGICAL_LAYOUT_UNKNOWN:
        return False
    if input_layout != output_layout:
        return False
    perm_values = [int(v) for v in list(perm)]
    if perm_values == list(range(rank)):
        return False
    permuted_input_shape = _permute_shape(input_shape, perm_values)
    if permuted_input_shape is None:
        return False
    # The metadata contract says the tensor stayed in the same known layout and
    # same shape. If the recorded permutation would produce a different shape,
    # the transpose is stale and must be elided.
    return permuted_input_shape != output_shape


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


def _read_onnx_squeeze_axes(node: Any) -> Optional[List[int]]:
    if node is None or str(getattr(node, "op_type", "")) != "Squeeze":
        return None
    for attribute in list(getattr(node, "attribute", [])):
        if str(getattr(attribute, "name", "")) == "axes":
            values = onnx.helper.get_attribute_value(attribute)
            if isinstance(values, (list, tuple)):
                return [int(v) for v in list(values)]
    return None


def _read_onnx_unsqueeze_axes(node: Any) -> Optional[List[int]]:
    if node is None or str(getattr(node, "op_type", "")) != "Unsqueeze":
        return None
    for attribute in list(getattr(node, "attribute", [])):
        if str(getattr(attribute, "name", "")) == "axes":
            values = onnx.helper.get_attribute_value(attribute)
            if isinstance(values, (list, tuple)):
                return [int(v) for v in list(values)]
    return None


def _compose_axis_permutations(
    first: Optional[Sequence[int]],
    second: Optional[Sequence[int]],
) -> Optional[List[int]]:
    if first is None and second is None:
        return None
    if first is None:
        composed = [int(v) for v in list(second or [])]
    elif second is None:
        composed = [int(v) for v in list(first)]
    else:
        first_values = [int(v) for v in list(first)]
        second_values = [int(v) for v in list(second)]
        if len(first_values) != len(second_values):
            return None
        if sorted(first_values) != list(range(len(first_values))):
            return None
        if sorted(second_values) != list(range(len(second_values))):
            return None
        composed = [int(first_values[int(idx)]) for idx in second_values]
    if composed == list(range(len(composed))):
        return None
    return composed


def _inverse_axis_permutation(perm: Optional[Sequence[int]]) -> Optional[List[int]]:
    if perm is None:
        return None
    values = [int(v) for v in list(perm)]
    if sorted(values) != list(range(len(values))):
        return None
    inverse = [0] * len(values)
    for new_axis, old_axis in enumerate(values):
        inverse[int(old_axis)] = int(new_axis)
    return inverse


def _constant_pad_pairs_for_tensor(tensor: Optional[TensorIR]) -> Optional[List[List[int]]]:
    if tensor is None or tensor.data is None:
        return None
    try:
        pads = np.asarray(tensor.data, dtype=np.int64).reshape(-1, 2)
    except Exception:
        return None
    return [[int(v) for v in list(row)] for row in pads.tolist()]


def _pad_output_matches_pre_permuted_input(
    *,
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
    pads_tensor: Optional[TensorIR],
    input_pre_permute: Optional[Sequence[int]],
) -> bool:
    if (
        input_tensor is None
        or output_tensor is None
        or pads_tensor is None
        or input_pre_permute is None
    ):
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    rank = len(input_shape)
    if rank == 0 or len(output_shape) != rank:
        return False
    inverse_perm = _inverse_axis_permutation(input_pre_permute)
    if inverse_perm is None or len(inverse_perm) != rank:
        return False
    pad_pairs = _constant_pad_pairs_for_tensor(pads_tensor)
    if pad_pairs is None:
        return False
    if len(pad_pairs) < rank:
        pad_pairs = ([[0, 0]] * (rank - len(pad_pairs))) + pad_pairs
    elif len(pad_pairs) > rank:
        pad_pairs = pad_pairs[-rank:]
    permuted_input_shape = _permute_shape(input_shape, inverse_perm)
    if permuted_input_shape is None or len(permuted_input_shape) != rank:
        return False
    padded_shape = [
        int(permuted_input_shape[idx]) + int(pad_pairs[idx][0]) + int(pad_pairs[idx][1])
        for idx in range(rank)
    ]
    return padded_shape == output_shape




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


def _should_emit_channel_last_space_to_depth(
    *,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    block_size: int,
) -> Optional[bool]:
    if len(list(input_shape)) != 4 or len(list(output_shape)) != 4:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    if 0 in {int(block_size)}:
        return None
    n, a, b, c = in_shape
    if a % block_size == 0 and b % block_size == 0:
        if out_shape == [n, a // block_size, b // block_size, c * block_size * block_size]:
            return True
    n, c, h, w = in_shape
    if h % block_size == 0 and w % block_size == 0:
        if out_shape == [n, c * block_size * block_size, h // block_size, w // block_size]:
            return False
    return None


def _should_emit_channel_last_depth_to_space(
    *,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    block_size: int,
) -> Optional[bool]:
    if len(list(input_shape)) != 4 or len(list(output_shape)) != 4:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    if 0 in {int(block_size)}:
        return None
    n, h, w, c = in_shape
    if c % (block_size * block_size) == 0:
        if out_shape == [n, h * block_size, w * block_size, c // (block_size * block_size)]:
            return True
    n, c, h, w = in_shape
    if c % (block_size * block_size) == 0:
        if out_shape == [n, c // (block_size * block_size), h * block_size, w * block_size]:
            return False
    return None


def _primary_data_input_name(op: OperatorIR) -> Optional[str]:
    op_type = str(op.op_type)
    if len(op.inputs) == 0:
        return None
    if op_type == "SPLIT":
        return str(op.inputs[1]) if len(op.inputs) >= 2 else str(op.inputs[0])
    if op_type == "SCATTER_ND":
        return str(op.inputs[1]) if len(op.inputs) >= 2 else None
    if op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"}:
        return str(op.inputs[2]) if len(op.inputs) >= 3 else None
    return str(op.inputs[0])


def _assign_tensor_logical_layout(
    tensor: Optional[TensorIR],
    layout: str,
) -> bool:
    if tensor is None:
        return False
    normalized_target = normalize_logical_layout(layout)
    if normalized_target == LOGICAL_LAYOUT_UNKNOWN:
        return False
    current_layout = normalize_logical_layout(tensor.logical_layout)
    if current_layout == normalized_target:
        return False
    if current_layout != LOGICAL_LAYOUT_UNKNOWN:
        current_rank = len(list(tensor.shape))
        current_is_channel_layout = (
            is_channel_first_logical_layout(current_layout)
            or is_channel_last_logical_layout(current_layout)
        )
        target_is_channel_layout = (
            is_channel_first_logical_layout(normalized_target)
            or is_channel_last_logical_layout(normalized_target)
        )
        if current_is_channel_layout and target_is_channel_layout:
            if current_rank != len(list(tensor.shape)):
                return False
    tensor.logical_layout = normalized_target
    return True


def _shared_tensor_layout(
    tensors: Sequence[Optional[TensorIR]],
) -> str:
    layouts: List[str] = []
    for tensor in tensors:
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        layout = normalize_logical_layout(tensor.logical_layout)
        if layout == LOGICAL_LAYOUT_UNKNOWN:
            return LOGICAL_LAYOUT_UNKNOWN
        if not (
            is_channel_first_logical_layout(layout)
            or is_channel_last_logical_layout(layout)
        ):
            return LOGICAL_LAYOUT_UNKNOWN
        layouts.append(layout)
    if len(layouts) == 0:
        return LOGICAL_LAYOUT_UNKNOWN
    first = layouts[0]
    if any(layout != first for layout in layouts[1:]):
        return LOGICAL_LAYOUT_UNKNOWN
    return first


def _infer_concat_peer_layout(
    op: OperatorIR,
    input_tensors: Sequence[Optional[TensorIR]],
) -> str:
    axis = op.options.get("axis", None)
    if axis is None:
        return LOGICAL_LAYOUT_UNKNOWN
    known_layout: Optional[str] = None
    known_rank: Optional[int] = None
    reference_shape: Optional[List[int]] = None
    for tensor in input_tensors:
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        layout = normalize_logical_layout(tensor.logical_layout)
        if layout == LOGICAL_LAYOUT_UNKNOWN:
            continue
        if not (
            is_channel_first_logical_layout(layout)
            or is_channel_last_logical_layout(layout)
        ):
            return LOGICAL_LAYOUT_UNKNOWN
        current_shape = [int(v) for v in list(tensor.shape)]
        if known_layout is None:
            known_layout = layout
            known_rank = rank
            reference_shape = current_shape
            continue
        if layout != known_layout or rank != known_rank:
            return LOGICAL_LAYOUT_UNKNOWN
        if reference_shape is not None:
            for dim_idx, (candidate_dim, expected_dim) in enumerate(zip(current_shape, reference_shape)):
                if int(dim_idx) == int(axis):
                    continue
                if int(candidate_dim) > 0 and int(expected_dim) > 0 and int(candidate_dim) != int(expected_dim):
                    return LOGICAL_LAYOUT_UNKNOWN
    if known_layout is None or known_rank is None:
        return LOGICAL_LAYOUT_UNKNOWN
    expected_axis = 1 if is_channel_first_logical_layout(known_layout) else int(known_rank) - 1
    if int(axis) != int(expected_axis):
        return LOGICAL_LAYOUT_UNKNOWN
    return str(known_layout)


def _can_emit_direct_torch_reshape_shape(
    shape_values: Sequence[int],
    *,
    allow_zero: bool,
) -> bool:
    values = [int(v) for v in list(shape_values)]
    if values.count(-1) > 1:
        return False
    for dim_value in values:
        if dim_value == -1:
            continue
        if dim_value == 0:
            if allow_zero:
                continue
            return False
        if dim_value < 0:
            return False
    return True


def _is_degenerate_sequence_like_rank4_or_rank5_tensor(
    tensor: Optional[TensorIR],
) -> bool:
    if tensor is None:
        return False
    shape_signature = (
        [int(v) for v in list(tensor.shape_signature)]
        if tensor.shape_signature is not None and len(list(tensor.shape_signature)) == len(list(tensor.shape))
        else [int(v) for v in list(tensor.shape)]
    )
    rank = len(shape_signature)
    if rank not in {4, 5}:
        return False
    if int(shape_signature[0]) not in {1, -1}:
        return False
    if any(int(dim) not in {1, -1} for dim in shape_signature[1:-1]):
        return False
    return int(shape_signature[-1]) > 0


def _is_channel_last_factorized_reshape(
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
) -> bool:
    if input_tensor is None or output_tensor is None:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    if not is_channel_last_logical_layout(input_layout):
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if len(input_shape) not in {3, 4, 5} or len(output_shape) not in {4, 5}:
        return False
    if len(output_shape) <= len(input_shape):
        return False
    if any(int(v) <= 0 for v in input_shape + output_shape):
        return False
    spatial_shape = input_shape[1:-1]
    spatial_rank = len(spatial_shape)
    if spatial_rank <= 0:
        return False
    if output_shape[0] != input_shape[0]:
        return False
    if output_shape[1:1 + spatial_rank] != spatial_shape:
        return False
    trailing_shape = output_shape[1 + spatial_rank:]
    if len(trailing_shape) < 2:
        return False
    return int(np.prod(trailing_shape, dtype=np.int64)) == int(input_shape[-1])


def _is_channel_last_factorized_rank3_sequence_reshape(
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
) -> bool:
    if input_tensor is None or output_tensor is None:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    if not is_channel_last_logical_layout(input_layout):
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if len(input_shape) not in {4, 5} or len(output_shape) != 3:
        return False
    if any(int(v) <= 0 for v in input_shape + output_shape):
        return False
    if int(output_shape[0]) != int(input_shape[0]):
        return False
    input_channels = int(input_shape[-1])
    output_features = int(output_shape[-1])
    if output_features <= 0 or input_channels <= 0 or input_channels % output_features != 0:
        return False
    spatial_extent = int(np.prod(input_shape[1:-1], dtype=np.int64))
    factor = int(input_channels // output_features)
    expected_sequence_extent = int(spatial_extent * factor)
    return int(output_shape[1]) == expected_sequence_extent


def _propagate_pytorch_friendly_layouts(model_ir: ModelIR) -> None:
    unary_passthrough_ops = {
        "ABS",
        "ATAN",
        "CEIL",
        "COS",
        "ELU",
        "EXP",
        "FLOOR",
        "HARD_SWISH",
        "IDENTITY",
        "LEAKY_RELU",
        "LOG",
        "LOGICAL_NOT",
        "LOGISTIC",
        "NEG",
        "RELU",
        "RELU6",
        "ROUND",
        "RSQRT",
        "SIGMOID",
        "SIGN",
        "SIN",
        "SQRT",
        "SQUARE",
        "TAN",
        "TANH",
    }
    binary_passthrough_ops = {
        "ADD",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "POW",
        "SUB",
    }
    resize_pool_passthrough_ops = {
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
    }
    changed = True
    while changed:
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            output_tensors = [
                model_ir.tensors.get(str(output_name), None)
                for output_name in op.outputs
            ]
            if op_type in unary_passthrough_ops and len(op.inputs) >= 1:
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(op.inputs[0]), None)]
                )
            elif op_type in binary_passthrough_ops and len(op.inputs) >= 2:
                propagated_layout = _shared_tensor_layout(
                    [
                        model_ir.tensors.get(str(op.inputs[0]), None),
                        model_ir.tensors.get(str(op.inputs[1]), None),
                    ]
                )
            elif op_type == "CONCATENATION":
                concat_input_tensors = [
                    model_ir.tensors.get(str(input_name), None) for input_name in op.inputs
                ]
                propagated_layout = _shared_tensor_layout(concat_input_tensors)
                if propagated_layout == LOGICAL_LAYOUT_UNKNOWN:
                    propagated_layout = _infer_concat_peer_layout(op, concat_input_tensors)
                    if propagated_layout != LOGICAL_LAYOUT_UNKNOWN:
                        for input_tensor in concat_input_tensors:
                            changed = _assign_tensor_logical_layout(input_tensor, propagated_layout) or changed
            elif op_type in {"PACK", "UNPACK"}:
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(input_name), None) for input_name in op.inputs]
                )
            elif op_type == "SPLIT":
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(op.inputs[-1]), None)]
                )
            elif op_type in resize_pool_passthrough_ops:
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(op.inputs[0]), None)]
                )
            else:
                continue
            if propagated_layout == LOGICAL_LAYOUT_UNKNOWN:
                continue
            for output_tensor in output_tensors:
                changed = _assign_tensor_logical_layout(output_tensor, propagated_layout) or changed


def _collect_feature_last_sequence_tensor_names(model_ir: ModelIR) -> Set[str]:
    consumers: Dict[str, List[int]] = {}
    producers: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_idx)

    def _is_time_major_recurrent_bridge(output_name: str) -> bool:
        for consumer_idx in consumers.get(str(output_name), []):
            consumer = model_ir.operators[int(consumer_idx)]
            if str(consumer.op_type) != "TRANSPOSE" or len(consumer.outputs) != 1:
                continue
            perm = _read_transpose_perm(model_ir, consumer)
            if perm != [1, 0, 2]:
                continue
            transpose_output_name = str(consumer.outputs[0])
            for next_idx in consumers.get(transpose_output_name, []):
                next_op_type = str(model_ir.operators[int(next_idx)].op_type)
                if next_op_type in {
                    "BIDIRECTIONAL_SEQUENCE_LSTM",
                    "UNIDIRECTIONAL_SEQUENCE_LSTM",
                    "UNIDIRECTIONAL_SEQUENCE_RNN",
                }:
                    return True
        return False

    def _trace_feature_last_rhs_seed(tensor_name: str) -> Optional[str]:
        visited: Set[str] = set()
        worklist: List[str] = [str(tensor_name)]
        passthrough_ops = {
            "CAST",
            "EXPAND_DIMS",
            "GATHER",
            "GATHER_ND",
            "IDENTITY",
            "RESHAPE",
            "SLICE",
            "SQUEEZE",
            "STRIDED_SLICE",
            "TRANSPOSE",
        }
        while len(worklist) > 0:
            current_name = str(worklist.pop())
            if current_name in visited:
                continue
            visited.add(current_name)
            current_tensor = model_ir.tensors.get(current_name, None)
            if current_tensor is not None:
                current_rank = len(list(current_tensor.shape))
                current_layout = normalize_logical_layout(current_tensor.logical_layout)
                if current_rank in {3, 4, 5} and is_channel_last_logical_layout(current_layout):
                    return current_name
            producer_idx = producers.get(current_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) not in passthrough_ops or len(producer.inputs) == 0:
                continue
            worklist.append(str(producer.inputs[0]))
        return None

    def _trace_feature_last_passthrough_inputs(tensor_name: str) -> Set[str]:
        traced_names: Set[str] = set()
        visited: Set[str] = set()
        worklist: List[str] = [str(tensor_name)]
        passthrough_ops = {
            "AVERAGE_POOL_2D",
            "CAST",
            "EXPAND_DIMS",
            "IDENTITY",
            "LEAKY_RELU",
            "LOGISTIC",
            "MAX_POOL_2D",
            "PAD",
            "PADV2",
            "RELU",
            "RELU6",
            "RESHAPE",
            "SQUEEZE",
            "STRIDED_SLICE",
            "TRANSPOSE",
        }
        while len(worklist) > 0:
            current_name = str(worklist.pop())
            if current_name in visited:
                continue
            visited.add(current_name)
            traced_names.add(current_name)
            producer_idx = producers.get(current_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) not in passthrough_ops or len(producer.inputs) == 0:
                continue
            upstream_name = str(producer.inputs[0])
            traced_names.add(upstream_name)
            worklist.append(upstream_name)
        return traced_names

    roots: Set[str] = set()
    if _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(model_ir):
        for tensor_name, tensor in model_ir.tensors.items():
            rank = len(list(tensor.shape))
            layout = normalize_logical_layout(tensor.logical_layout)
            if rank in {4, 5} and is_channel_last_logical_layout(layout):
                roots.add(str(tensor_name))
    for tensor_name, tensor in model_ir.tensors.items():
        normalized_name = str(tensor_name)
        rank = len(list(tensor.shape))
        layout = normalize_logical_layout(tensor.logical_layout)
        lowered_name = normalized_name.lower()
        if (
            rank in {3, 4, 5}
            and is_channel_last_logical_layout(layout)
            and any(token in lowered_name for token in ("_nwc", "_nhwc", "_ndhwc"))
        ):
            roots.add(normalized_name)
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type == "BATCH_MATMUL" and len(op.inputs) >= 2:
            rhs_seed = _trace_feature_last_rhs_seed(str(op.inputs[1]))
            if rhs_seed is not None:
                roots.add(rhs_seed)
        if op_type == "TRANSPOSE" and len(op.inputs) >= 1 and len(op.outputs) == 1:
            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            output_tensor = model_ir.tensors.get(output_name, None)
            input_tensor = model_ir.tensors.get(input_name, None)
            if output_tensor is None:
                continue
            rank = len(list(output_tensor.shape))
            if rank != 3:
                continue
            perm = _read_transpose_perm(model_ir, op)
            if (
                perm == [1, 0, 2]
                and input_tensor is not None
                and (
                    is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                    or is_channel_last_logical_layout(normalize_logical_layout(output_tensor.logical_layout))
                )
            ):
                roots.add(input_name)
                roots.add(output_name)
                continue
            if perm != _perm_cf_to_cl(rank):
                continue
            producer_idx = producers.get(input_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) != "RESHAPE" or len(producer.outputs) != 1:
                continue
            roots.add(output_name)
            continue
        if op_type != "RESHAPE" or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None:
            continue
        rank = len(list(output_tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None
        if (
            output_name in set(str(v) for v in model_ir.outputs)
            and input_tensor is not None
            and len(list(input_tensor.shape)) >= rank
            and len(list(output_tensor.shape)) >= 1
            and len(list(input_tensor.shape)) >= 1
            and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
        ):
            roots.add(output_name)
            continue
        if (
            input_tensor is not None
            and len(list(input_tensor.shape)) >= rank
            and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
            and len(list(output_tensor.shape)) >= 1
            and len(list(input_tensor.shape)) >= 1
            and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
        ):
            roots.add(output_name)
            continue
        if _is_channel_last_factorized_reshape(input_tensor, output_tensor):
            roots.add(output_name)
            continue
        if _is_channel_last_factorized_rank3_sequence_reshape(input_tensor, output_tensor):
            roots.add(output_name)
            continue
        if input_tensor is not None and rank == 3 and len(list(input_tensor.shape)) == 3:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if (
                int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
            ):
                for consumer_idx in consumers.get(output_name, []):
                    consumer = model_ir.operators[int(consumer_idx)]
                    if (
                        str(consumer.op_type) != "BATCH_MATMUL"
                        or len(consumer.inputs) < 2
                        or str(consumer.inputs[0]) != output_name
                        or not bool(consumer.options.get("adjX", False))
                    ):
                        continue
                    rhs_tensor = model_ir.tensors.get(str(consumer.inputs[1]), None)
                    if rhs_tensor is None or len(list(rhs_tensor.shape)) < 2:
                        continue
                    rhs_contract = int(list(rhs_tensor.shape)[-2])
                    if rhs_contract != int(input_shape[-1]):
                        continue
                    roots.add(output_name)
                    break
                if output_name in roots:
                    continue
        if input_tensor is not None and rank == 3 and len(list(input_tensor.shape)) in {4, 5}:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if (
                int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
                and _is_time_major_recurrent_bridge(output_name)
            ):
                roots.add(output_name)
                input_name = str(op.inputs[0])
                roots.add(input_name)
                producer: Optional[OperatorIR] = None
                producer_output_name = ""
                producer_rank = -1
                producer_idx = producers.get(input_name, None)
                if producer_idx is not None:
                    producer = model_ir.operators[int(producer_idx)]
                    producer_output_name = str(producer.outputs[0]) if len(producer.outputs) == 1 else ""
                    producer_output_tensor = (
                        model_ir.tensors.get(producer_output_name, None)
                        if producer_output_name != ""
                        else None
                    )
                    producer_rank = (
                        len(list(producer_output_tensor.shape))
                        if producer_output_tensor is not None
                        else -1
                    )
                if (
                    producer is not None
                    and str(producer.op_type) == "TRANSPOSE"
                    and producer_rank in {4, 5}
                ):
                    if len(producer.inputs) >= 1:
                        producer_input_name = str(producer.inputs[0])
                        roots.update(_trace_feature_last_passthrough_inputs(producer_input_name))
                    roots.add(producer_output_name)
                continue
            if (
                int(np.prod(input_shape[1:], dtype=np.int64))
                == int(np.prod(output_shape[1:], dtype=np.int64))
                and (
                    is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                    or int(input_shape[-1]) == 1
                )
            ):
                for consumer_idx in consumers.get(output_name, []):
                    consumer = model_ir.operators[int(consumer_idx)]
                    if (
                        str(consumer.op_type) != "BATCH_MATMUL"
                        or len(consumer.inputs) < 2
                        or str(consumer.inputs[1]) != output_name
                    ):
                        continue
                    lhs_tensor = model_ir.tensors.get(str(consumer.inputs[0]), None)
                    if lhs_tensor is None or len(list(lhs_tensor.shape)) < 2:
                        continue
                    lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
                    if int(lhs_shape[-1]) != int(output_shape[-2]):
                        continue
                    roots.add(output_name)
                    break
                if output_name in roots:
                    continue
        raw_shape = op.options.get("onnxRawNewShape", None)
        new_shape = op.options.get("newShape", None)
        if not isinstance(raw_shape, list) or not isinstance(new_shape, list):
            continue
        raw_shape_values = [int(v) for v in list(raw_shape)]
        new_shape_values = [int(v) for v in list(new_shape)]
        if raw_shape_values == new_shape_values:
            continue
        if len(raw_shape_values) != rank or len(new_shape_values) != rank:
            continue
        if consumers.get(output_name):
            if any(
                str(model_ir.operators[int(consumer_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(consumer_idx)]) == _perm_cf_to_cl(rank)
                for consumer_idx in consumers.get(output_name, [])
            ):
                continue
        roots.add(output_name)

    preserve_names: Set[str] = set(roots)
    if len(roots) == 0:
        return preserve_names

    layout_passthrough_ops = {
        "ABS",
        "ADD",
        "AVERAGE_POOL_2D",
        "ATAN",
        "BATCH_MATMUL",
        "BROADCAST_TO",
        "CAST",
        "CEIL",
        "CONCATENATION",
        "COS",
        "DEPTH_TO_SPACE",
        "DIV",
        "ELU",
        "ERF",
        "EXP",
        "EXPAND_DIMS",
        "GATHER",
        "GATHER_ND",
        "GELU",
        "IDENTITY",
        "LEAKY_RELU",
        "LOG",
        "LOGISTIC",
        "MATMUL",
        "MAXIMUM",
        "MEAN",
        "MINIMUM",
        "MUL",
        "MAX_POOL_2D",
        "NEG",
        "PACK",
        "POW",
        "RELU",
        "RELU6",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SIGMOID",
        "SIGN",
        "SIN",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQRT",
        "SQUARE",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "SUM",
        "TANH",
        "TILE",
        "TRANSPOSE",
        "UNPACK",
    }
    changed = True
    while changed:
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            if op_type not in layout_passthrough_ops:
                continue
            input_names = [str(v) for v in op.inputs]
            output_names = [str(v) for v in op.outputs]
            if len(output_names) == 0:
                continue
            has_preserved_input = any(name in preserve_names for name in input_names)
            has_preserved_output = any(name in preserve_names for name in output_names)
            if not has_preserved_input and not has_preserved_output:
                continue
            if has_preserved_input:
                if op_type != "TRANSPOSE" or len(op.outputs) != 1:
                    for output_name in output_names:
                        if output_name not in preserve_names:
                            preserve_names.add(output_name)
                            changed = True
                else:
                    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                    rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
                    perm = _read_transpose_perm(model_ir, op)
                    if not (
                        rank in {3, 4, 5}
                        and (
                            perm == _perm_cl_to_cf(rank)
                            or perm == _perm_cf_to_cl(rank)
                        )
                    ):
                        for output_name in output_names:
                            if output_name not in preserve_names:
                                preserve_names.add(output_name)
                                changed = True
            if has_preserved_output:
                if (
                    op_type == "RESHAPE"
                    and len(op.inputs) >= 1
                    and len(op.outputs) == 1
                    and _is_channel_last_factorized_rank3_sequence_reshape(
                        model_ir.tensors.get(str(op.inputs[0]), None),
                        model_ir.tensors.get(str(op.outputs[0]), None),
                    )
                ):
                    continue
                if op_type != "TRANSPOSE" or len(op.inputs) < 1:
                    for input_name in input_names:
                        if input_name not in preserve_names:
                            preserve_names.add(input_name)
                            changed = True
                else:
                    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                    rank = len(list(input_tensor.shape)) if input_tensor is not None else -1
                    perm = _read_transpose_perm(model_ir, op)
                    if not (
                        rank in {3, 4, 5}
                        and (
                            perm == _perm_cl_to_cf(rank)
                            or perm == _perm_cf_to_cl(rank)
                        )
                    ):
                        for input_name in input_names:
                            if input_name not in preserve_names:
                                preserve_names.add(input_name)
                                changed = True
    return preserve_names


def _apply_feature_last_sequence_layouts(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    if len(preserve_channel_last_tensor_names) == 0:
        return
    producers: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_idx)
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))
    for tensor_name in preserve_channel_last_tensor_names:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        if is_channel_last_logical_layout(
            normalize_logical_layout(tensor.logical_layout)
        ):
            continue
        tensor.logical_layout = LOGICAL_LAYOUT_UNKNOWN

    for op in model_ir.operators:
        output_name = str(op.outputs[0]) if len(op.outputs) == 1 else None
        if output_name is None or output_name not in preserve_channel_last_tensor_names:
            continue
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None:
            continue
        rank = len(list(output_tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        op_type = str(op.op_type)
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None
        if op_type == "TRANSPOSE":
            perm = _read_transpose_perm(model_ir, op)
            if output_name in preserve_channel_last_tensor_names and rank == 3 and perm == [1, 0, 2]:
                output_tensor.logical_layout = channel_last_logical_layout(rank)
                continue
            if perm == _perm_cf_to_cl(rank):
                input_layout = normalize_logical_layout(
                    input_tensor.logical_layout if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
                )
                if (
                    rank == 3
                    and output_name in set(str(v) for v in model_ir.outputs)
                    and output_name not in preserve_channel_last_tensor_names
                    and is_channel_last_logical_layout(input_layout)
                ):
                    output_tensor.logical_layout = channel_first_logical_layout(rank)
                else:
                    output_tensor.logical_layout = channel_last_logical_layout(rank)
            elif rank in {4, 5}:
                output_tensor.logical_layout = channel_last_logical_layout(rank)
            elif (
                input_tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                and isinstance(perm, list)
                and len(perm) == rank
                and sorted(int(v) for v in perm) == list(range(rank))
                and int(perm[0]) == 0
                and int(perm[-1]) == rank - 1
            ):
                output_tensor.logical_layout = channel_last_logical_layout(rank)
            continue
        if op_type == "RESHAPE":
            should_mark_channel_last = False
            if output_name in preserve_channel_last_tensor_names and rank == 3:
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and output_name in set(str(v) for v in model_ir.outputs)
                and input_tensor is not None
                and len(list(input_tensor.shape)) >= rank
                and len(list(output_tensor.shape)) >= 1
                and len(list(input_tensor.shape)) >= 1
                and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
            ):
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and input_tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
            ):
                should_mark_channel_last = True
            raw_shape = op.options.get("onnxRawNewShape", None)
            if not should_mark_channel_last and isinstance(raw_shape, list):
                raw_shape_values = [int(v) for v in list(raw_shape)]
                if len(raw_shape_values) == rank:
                    current_shape = [int(v) for v in list(output_tensor.shape)]
                    if raw_shape_values != current_shape and raw_shape_values[-1] == current_shape[-1]:
                        should_mark_channel_last = True
            if not should_mark_channel_last and _is_channel_last_factorized_reshape(input_tensor, output_tensor):
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and _is_channel_last_factorized_rank3_sequence_reshape(input_tensor, output_tensor)
            ):
                should_mark_channel_last = True
            if should_mark_channel_last:
                output_tensor.logical_layout = channel_last_logical_layout(rank)
            continue

    safe_passthrough_ops = {
        "ABS",
        "ADD",
        "ATAN",
        "AVERAGE_POOL_2D",
        "BATCH_MATMUL",
        "CAST",
        "CONCATENATION",
        "DEPTH_TO_SPACE",
        "DIV",
        "ELU",
        "ERF",
        "EXP",
        "EXPAND_DIMS",
        "GELU",
        "IDENTITY",
        "LOGISTIC",
        "MAXIMUM",
        "MAX_POOL_2D",
        "MEAN",
        "MINIMUM",
        "MUL",
        "NEG",
        "PACK",
        "RELU",
        "RELU6",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SIGMOID",
        "SIGN",
        "SIN",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQRT",
        "SQUARE",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "SUM",
        "TANH",
        "TILE",
        "UNPACK",
    }
    changed = True
    while changed:
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            if op_type not in safe_passthrough_ops or len(op.outputs) == 0:
                continue
            input_tensors = [model_ir.tensors.get(str(name), None) for name in op.inputs]
            if not any(
                tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout))
                for tensor in input_tensors
            ):
                continue
            for output_name in op.outputs:
                output_tensor = model_ir.tensors.get(str(output_name), None)
                if output_tensor is None:
                    continue
                rank = len(list(output_tensor.shape))
                if rank not in {3, 4, 5}:
                    continue
                target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    changed = True
    for op in model_ir.operators:
        if str(op.op_type) != "RESHAPE" or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        if output_name not in preserve_channel_last_tensor_names:
            continue
        if consumers.get(output_name):
            if any(
                str(model_ir.operators[int(consumer_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(consumer_idx)])
                == _perm_cf_to_cl(len(list(model_ir.tensors[output_name].shape)))
                for consumer_idx in consumers.get(output_name, [])
            ):
                continue
        raw_shape = op.options.get("onnxRawNewShape", None)
        if not isinstance(raw_shape, list):
            continue
        raw_shape_values = [int(v) for v in list(raw_shape)]
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None or len(raw_shape_values) != len(list(output_tensor.shape)):
            continue
        output_tensor.shape = list(raw_shape_values)
        output_tensor.shape_signature = list(raw_shape_values)
        op.options["newShape"] = list(raw_shape_values)
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if shape_tensor is not None and isinstance(shape_tensor.data, np.ndarray):
                dtype = np.asarray(shape_tensor.data).dtype
                shape_tensor.data = np.asarray(raw_shape_values, dtype=dtype)
                shape_tensor.shape = [int(len(raw_shape_values))]
                shape_tensor.shape_signature = [int(len(raw_shape_values))]


def _rewrite_layout_sensitive_ops(
    model_ir: ModelIR,
    original_layouts: Dict[str, str],
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    rewritten_constant_tensor_names: Set[str] = set()

    def _rewrite_tensor_once(
        tensor_name: str,
        rewrite_fn: Callable[[], bool],
    ) -> None:
        if str(tensor_name) in rewritten_constant_tensor_names:
            return
        if rewrite_fn():
            rewritten_constant_tensor_names.add(str(tensor_name))

    for op in model_ir.operators:
        op_type = str(op.op_type)
        data_input_name = _primary_data_input_name(op)
        data_tensor = model_ir.tensors.get(str(data_input_name), None) if data_input_name is not None else None
        if data_tensor is None:
            continue
        if any(str(name) in preserve_channel_last_tensor_names for name in list(op.inputs) + list(op.outputs)):
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
            if op_type in {"ARG_MAX", "ARG_MIN"} and len(op.inputs) >= 2:
                axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if axis_tensor is not None:
                    resolved_axis_tensor = axis_tensor
                    _rewrite_tensor_once(
                        str(op.inputs[1]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=resolved_axis_tensor,
                            source_layout=original_layout,
                            target_layout=target_layout,
                            rank=rank,
                        ),
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
                    resolved_axis_tensor = axis_tensor
                    _rewrite_tensor_once(
                        str(op.inputs[0]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=resolved_axis_tensor,
                            source_layout=original_layout,
                            target_layout=target_layout,
                            rank=rank,
                        ),
                    )
        elif op_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            if len(op.inputs) >= 2:
                axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if axis_tensor is not None:
                    resolved_axis_tensor = axis_tensor
                    _rewrite_tensor_once(
                        str(op.inputs[1]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=resolved_axis_tensor,
                            source_layout=original_layout,
                            target_layout=target_layout,
                            rank=rank,
                        ),
                    )
        elif op_type in {"SLICE", "STRIDED_SLICE"}:
            for input_name in op.inputs[1:4]:
                vector_tensor = model_ir.tensors.get(str(input_name), None)
                if vector_tensor is not None:
                    _rewrite_tensor_once(
                        str(input_name),
                        lambda vector_tensor=vector_tensor: _rewrite_vector_constant_inplace(
                            tensor=vector_tensor,
                            perm=logical_layout_permutation(
                                source_layout=original_layout,
                                target_layout=target_layout,
                            ) or [],
                            expected_rank=rank,
                        ),
                    )
        elif op_type in {"PAD", "PADV2", "MIRROR_PAD"} and len(op.inputs) >= 2:
            pad_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if pad_tensor is not None:
                _rewrite_tensor_once(
                    str(op.inputs[1]),
                    lambda pad_tensor=pad_tensor: _rewrite_matrix_constant_inplace(
                        tensor=pad_tensor,
                        perm=logical_layout_permutation(
                            source_layout=original_layout,
                            target_layout=target_layout,
                        ) or [],
                        expected_rank=rank,
                    ),
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
                    resolved_perm_tensor = perm_tensor
                    perm_tensor_dtype = np.asarray(resolved_perm_tensor.data).dtype
                    perm_values = [int(v) for v in np.asarray(resolved_perm_tensor.data).reshape(-1).tolist()]
                    if len(perm_values) == rank:
                        def _rewrite_perm_tensor() -> bool:
                            rewritten_perm = [int(old_axis_to_new_axis[int(axis)]) for axis in perm_values]
                            resolved_perm_tensor.data = np.asarray(rewritten_perm, dtype=perm_tensor_dtype)
                            resolved_perm_tensor.shape = [int(rank)]
                            resolved_perm_tensor.shape_signature = [int(rank)]
                            return True
                        _rewrite_tensor_once(str(op.inputs[1]), _rewrite_perm_tensor)
            elif "perm" in op.options:
                perm_values = [int(v) for v in list(op.options.get("perm", []))]
                if len(perm_values) == rank:
                    op.options["perm"] = [int(old_axis_to_new_axis[int(axis)]) for axis in perm_values]
        elif op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} and len(op.inputs) >= 1:
            output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            if output_shape_tensor is not None:
                _rewrite_tensor_once(
                    str(op.inputs[0]),
                    lambda output_shape_tensor=output_shape_tensor: _rewrite_vector_constant_inplace(
                        tensor=output_shape_tensor,
                        perm=logical_layout_permutation(
                            source_layout=original_layout,
                            target_layout=target_layout,
                        ) or [],
                        expected_rank=rank,
                    ),
                )
        elif op_type == "RESHAPE" and len(op.outputs) == 1:
            out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            if out_tensor is not None:
                preferred_shape = _preferred_reshape_target_values(out_tensor)
                if preferred_shape is None:
                    preferred_shape = [int(v) for v in list(out_tensor.shape)]
                resolved_preferred_shape = [int(v) for v in list(preferred_shape)]
                if len(op.inputs) >= 2:
                    shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                    if shape_tensor is not None and isinstance(shape_tensor.data, np.ndarray):
                        resolved_shape_tensor = shape_tensor
                        shape_tensor_dtype = np.asarray(resolved_shape_tensor.data).dtype
                        def _rewrite_shape_tensor() -> bool:
                            resolved_shape_tensor.data = np.asarray(resolved_preferred_shape, dtype=shape_tensor_dtype)
                            resolved_shape_tensor.shape = [int(len(resolved_preferred_shape))]
                            resolved_shape_tensor.shape_signature = [int(len(resolved_preferred_shape))]
                            return True
                        _rewrite_tensor_once(str(op.inputs[1]), _rewrite_shape_tensor)
                op.options["newShape"] = list(resolved_preferred_shape)


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
        elif op_type in {"CONV_3D", "CONV_3D_TRANSPOSE"} and arr.ndim == 5:
            if is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout)):
                tensor.data = np.transpose(arr, (4, 3, 0, 1, 2)).copy()
            else:
                continue
        else:
            continue
        tensor.shape = [int(v) for v in list(tensor.data.shape)]
        if tensor.shape_signature is not None and len(list(tensor.shape_signature)) == int(arr.ndim):
            tensor.shape_signature = [int(v) for v in list(tensor.shape)]
        rewritten_weights.add(weight_name)


def _synchronize_reshape_targets_with_output_tensors(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    for op in model_ir.operators:
        if str(op.op_type) != "RESHAPE" or len(op.outputs) != 1:
            continue
        if str(op.outputs[0]) in preserve_channel_last_tensor_names:
            continue
        out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if out_tensor is None:
            continue
        preferred_shape = _preferred_reshape_target_values(out_tensor)
        if preferred_shape is None or len(preferred_shape) == 0:
            continue
        op.options["newShape"] = list(preferred_shape)
        if len(op.inputs) < 2:
            continue
        shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        if shape_tensor is None or not isinstance(shape_tensor.data, np.ndarray):
            continue
        dtype = np.asarray(shape_tensor.data).dtype
        shape_tensor.data = np.asarray(preferred_shape, dtype=dtype)
        shape_tensor.shape = [int(len(preferred_shape))]
        shape_tensor.shape_signature = [int(len(preferred_shape))]


def _remove_redundant_layout_transposes(
    model_ir: ModelIR,
    original_layouts: Dict[str, str],
    preserve_channel_last_tensor_names: Set[str],
) -> None:
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
        if input_name in preserve_channel_last_tensor_names or output_name in preserve_channel_last_tensor_names:
            continue
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        reference_tensor = output_tensor if output_tensor is not None else input_tensor
        rank = len(list(reference_tensor.shape)) if reference_tensor is not None else -1
        if rank not in {3, 4, 5}:
            continue
        consumer_op_types = {
            str(model_ir.operators[int(consumer_idx)].op_type)
            for consumer_idx in consumers.get(output_name, [])
            if int(consumer_idx) != int(op_idx)
        }
        reshape_only_consumers = len(consumer_op_types) > 0 and consumer_op_types == {"RESHAPE"}
        if (
            reshape_only_consumers
            and input_tensor is not None
            and output_tensor is not None
            and [int(v) for v in list(input_tensor.shape)] != [int(v) for v in list(output_tensor.shape)]
        ):
            continue
        if len(consumer_op_types & {"GATHER", "GATHER_ND", "SLICE", "STRIDED_SLICE"}) > 0:
            continue
        perm = _read_transpose_perm(model_ir, op)
        input_layout = normalize_logical_layout(original_layouts.get(input_name, LOGICAL_LAYOUT_UNKNOWN))
        output_layout = normalize_logical_layout(original_layouts.get(output_name, LOGICAL_LAYOUT_UNKNOWN))
        remove_as_identity = bool(
            perm is not None
            and (
                perm == list(range(rank))
                or _is_layout_only_transpose_by_shape(
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    perm=perm,
                )
                and _is_standard_channel_layout_permutation(
                    perm=perm,
                    rank=rank,
                )
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
                or _is_inconsistent_standard_layout_transpose(
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    perm=perm,
                ) and not reshape_only_consumers
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
                "IDENTITY",
                [input_name],
                [output_name],
                {},
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


def _rewrite_atan2_ones_like_to_atan(model_ir: ModelIR) -> None:
    for op in model_ir.operators:
        if str(op.op_type) != "ATAN2" or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue
        lhs_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        rhs_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if (
            lhs_tensor is None
            or rhs_tensor is None
            or out_tensor is None
            or not isinstance(rhs_tensor.data, np.ndarray)
        ):
            continue
        rhs_values = np.asarray(rhs_tensor.data)
        if rhs_values.size == 0 or not np.allclose(rhs_values, 1.0):
            continue
        lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
        out_shape = [int(v) for v in list(out_tensor.shape)]
        if lhs_shape != out_shape:
            continue
        rhs_shape = [int(v) for v in list(rhs_tensor.shape)]
        if rhs_shape != lhs_shape:
            perm = _perm_cl_to_cf(len(rhs_shape))
            perm_inv = _perm_cf_to_cl(len(rhs_shape))
            if (
                perm is None
                or _permute_shape(rhs_shape, perm) != lhs_shape
            ) and (
                perm_inv is None
                or _permute_shape(rhs_shape, perm_inv) != lhs_shape
            ):
                continue
        op.op_type = "ATAN"
        op.inputs = [str(op.inputs[0])]


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


def _reject_residual_layout_transposes(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    consumers: Dict[str, List[int]] = {}
    public_layout_bridge_tensor_names = model_ir.metadata.get("public_layout_bridge_tensor_names", [])
    if not isinstance(public_layout_bridge_tensor_names, list):
        public_layout_bridge_tensor_names = []
    public_layout_bridge_tensor_name_set = {str(name) for name in public_layout_bridge_tensor_names}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    for op in model_ir.operators:
        if str(op.op_type) != "TRANSPOSE":
            continue
        related_tensor_names = [str(v) for v in list(op.inputs) + list(op.outputs)]
        if any(name in public_layout_bridge_tensor_name_set for name in related_tensor_names):
            continue
        if any(name in preserve_channel_last_tensor_names for name in related_tensor_names):
            continue
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
        transpose_consumer_indices = [int(v) for v in consumers.get(output_name, [])]
        if (
            len(transpose_consumer_indices) > 0
            and all(
                str(model_ir.operators[int(consumer_idx)].op_type) == "RESHAPE"
                for consumer_idx in transpose_consumer_indices
            )
        ):
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
        if _is_layout_only_transpose_by_shape(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            perm=perm,
        ):
            continue
        if _is_reshape_only_residual_layout_bridge_transpose(
            model_ir=model_ir,
            op=op,
            consumers=consumers,
        ):
            continue
        if perm == _perm_cl_to_cf(rank) or perm == _perm_cf_to_cl(rank):
            raise ModelIRPyTorchExportError(
                "Channel-first normalization failed: residual layout transpose remains. "
                f"op_type={op.op_type} outputs={op.outputs} perm={perm}"
            )


def _is_reshape_only_residual_layout_bridge_transpose(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    consumers: Optional[Dict[str, List[int]]] = None,
) -> bool:
    if str(op.op_type) != "TRANSPOSE":
        return False
    output_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
    output_tensor = model_ir.tensors.get(output_name, None)
    rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
    if rank not in {3, 4, 5}:
        return False
    input_name = str(op.inputs[0]) if len(op.inputs) > 0 else ""
    input_tensor = model_ir.tensors.get(input_name, None)
    if input_tensor is None or output_tensor is None:
        return False
    perm = _read_transpose_perm(model_ir, op)
    if perm != _perm_cl_to_cf(rank) and perm != _perm_cf_to_cl(rank):
        return False
    if [int(v) for v in list(input_tensor.shape)] != [int(v) for v in list(output_tensor.shape)]:
        return False
    if normalize_logical_layout(input_tensor.logical_layout) != normalize_logical_layout(output_tensor.logical_layout):
        return False
    if consumers is None:
        consumers = {}
        for op_idx, candidate in enumerate(model_ir.operators):
            for candidate_input_name in candidate.inputs:
                consumers.setdefault(str(candidate_input_name), []).append(int(op_idx))
    user_indices = [int(v) for v in consumers.get(output_name, [])]
    return len(user_indices) > 0 and all(
        str(model_ir.operators[int(user_idx)].op_type) == "RESHAPE"
        for user_idx in user_indices
    )


def _align_public_boundary_shapes_to_onnx_contract(model_ir: ModelIR) -> None:
    boundary_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(boundary_map, dict):
        boundary_map = {}
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
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        desired_layout = normalize_logical_layout(
            public_layout_map.get(
                str(tensor_name),
                channel_last_logical_layout(rank) if recurrent_sequence_context and rank == 3 else channel_first_logical_layout(rank),
            )
        ) if rank in {3, 4, 5} else LOGICAL_LAYOUT_UNKNOWN
        current_layout = normalize_logical_layout(tensor.logical_layout)
        if isinstance(boundary_shape, list) and len(boundary_shape) == rank:
            tensor.shape_signature = [int(v) for v in list(boundary_shape)]
            tensor.shape = [int(v) if int(v) > 0 else 1 for v in list(boundary_shape)]
        elif (
            rank in {3, 4, 5}
            and desired_layout != LOGICAL_LAYOUT_UNKNOWN
            and current_layout != LOGICAL_LAYOUT_UNKNOWN
            and desired_layout != current_layout
        ):
            perm_to_public = logical_layout_permutation(
                source_layout=current_layout,
                target_layout=desired_layout,
            )
            current_shape_signature = list(tensor.shape_signature or tensor.shape)
            permuted_shape = (
                None
                if perm_to_public is None
                else _permute_shape(current_shape_signature, perm_to_public)
            )
            if permuted_shape is not None:
                tensor.shape_signature = [int(v) for v in list(permuted_shape)]
                tensor.shape = [int(v) if int(v) > 0 else 1 for v in list(permuted_shape)]
        if rank in {3, 4, 5}:
            tensor.logical_layout = desired_layout


def _ensure_public_boundary_layout_bridges(
    *,
    model_ir: ModelIR,
    desired_public_shape_map: Dict[str, List[int]],
    desired_public_layout_map: Dict[str, str],
) -> None:
    used_tensor_names: Set[str] = set(model_ir.tensors.keys())
    bridge_tensor_names = model_ir.metadata.get("public_layout_bridge_tensor_names", [])
    if not isinstance(bridge_tensor_names, list):
        bridge_tensor_names = []
    model_ir.metadata["public_layout_bridge_tensor_names"] = bridge_tensor_names

    def _insert_public_boundary_layout_bridge(
        *,
        tensor_name: str,
        current_tensor: TensorIR,
        desired_shape: Sequence[int],
        desired_layout: str,
        is_input: bool,
    ) -> None:
        current_shape = [int(v) for v in list(current_tensor.shape_signature or current_tensor.shape)]
        target_shape = [int(v) for v in list(desired_shape)]
        current_layout = normalize_logical_layout(current_tensor.logical_layout)
        normalized_target_layout = normalize_logical_layout(desired_layout)
        if (
            len(current_shape) not in {3, 4, 5}
            or len(current_shape) != len(target_shape)
            or current_layout == LOGICAL_LAYOUT_UNKNOWN
            or normalized_target_layout == LOGICAL_LAYOUT_UNKNOWN
            or current_layout == normalized_target_layout
        ):
            return
        perm = logical_layout_permutation(
            source_layout=normalized_target_layout if is_input else current_layout,
            target_layout=current_layout if is_input else normalized_target_layout,
        )
        expected_shape = current_shape if is_input else target_shape
        seed_shape = target_shape if is_input else current_shape
        if perm is None or _permute_shape(seed_shape, perm) != expected_shape:
            return
        bridge_tensor_name = _make_unique_identifier(
            f"{tensor_name}_public_layout_bridge",
            used_tensor_names,
        )
        bridge_tensor = _clone_tensor(current_tensor)
        bridge_tensor.name = str(bridge_tensor_name)
        model_ir.tensors[str(bridge_tensor_name)] = bridge_tensor
        if str(bridge_tensor_name) not in bridge_tensor_names:
            bridge_tensor_names.append(str(bridge_tensor_name))
        perm_name = _make_unique_identifier(
            f"{bridge_tensor_name}_perm",
            used_tensor_names,
        )
        perm_arr = np.asarray([int(v) for v in list(perm)], dtype=np.int32)
        model_ir.tensors[str(perm_name)] = TensorIR(
            name=str(perm_name),
            dtype="INT32",
            shape=[int(perm_arr.size)],
            shape_signature=[int(perm_arr.size)],
            data=perm_arr,
        )
        if is_input:
            for op in model_ir.operators:
                op.inputs = [
                    str(bridge_tensor_name) if str(name) == str(tensor_name) else str(name)
                    for name in list(op.inputs)
                ]
            model_ir.operators.insert(
                0,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[str(tensor_name), str(perm_name)],
                    outputs=[str(bridge_tensor_name)],
                    options={"perm": [int(v) for v in list(perm)]},
                ),
            )
            return
        for op in model_ir.operators:
            op.outputs = [
                str(bridge_tensor_name) if str(name) == str(tensor_name) else str(name)
                for name in list(op.outputs)
            ]
            op.inputs = [
                str(bridge_tensor_name) if str(name) == str(tensor_name) else str(name)
                for name in list(op.inputs)
            ]
        model_ir.operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[str(bridge_tensor_name), str(perm_name)],
                outputs=[str(tensor_name)],
                options={"perm": [int(v) for v in list(perm)]},
            )
        )

    for tensor_name in list(model_ir.inputs):
        current_tensor = model_ir.tensors.get(str(tensor_name), None)
        desired_shape = desired_public_shape_map.get(str(tensor_name), None)
        desired_layout = desired_public_layout_map.get(str(tensor_name), LOGICAL_LAYOUT_UNKNOWN)
        if current_tensor is None or desired_shape is None:
            continue
        _insert_public_boundary_layout_bridge(
            tensor_name=str(tensor_name),
            current_tensor=current_tensor,
            desired_shape=desired_shape,
            desired_layout=desired_layout,
            is_input=True,
        )

    for tensor_name in list(model_ir.outputs):
        current_tensor = model_ir.tensors.get(str(tensor_name), None)
        desired_shape = desired_public_shape_map.get(str(tensor_name), None)
        desired_layout = desired_public_layout_map.get(str(tensor_name), LOGICAL_LAYOUT_UNKNOWN)
        if current_tensor is None or desired_shape is None:
            continue
        _insert_public_boundary_layout_bridge(
            tensor_name=str(tensor_name),
            current_tensor=current_tensor,
            desired_shape=desired_shape,
            desired_layout=desired_layout,
            is_input=False,
        )


def validate_channel_first_exportability(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
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
    public_layout_bridge_tensor_names = set(
        str(name)
        for name in list(model_ir.metadata.get("public_layout_bridge_tensor_names", []))
    )
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
            if str(tensor_name) in preserve_channel_last_tensor_names:
                continue
            if str(tensor_name) in public_layout_bridge_tensor_names:
                continue
            if recurrent_sequence_context and op_type in {"CONCATENATION", "SLICE", "STRIDED_SLICE", "SPLIT"}:
                continue
            layout = normalize_logical_layout(tensor.logical_layout)
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and op_type == "SCATTER_ND"
                and primary_name is not None
                and str(tensor_name) == str(primary_name)
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and op_type == "SCATTER_ND"
                and str(tensor_name) in {str(v) for v in list(op.outputs)}
                and str(tensor_name) not in {str(v) for v in list(model_ir.outputs)}
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and rank in {3, 5}
                and op_type in {"CONCATENATION", "GATHER", "GATHER_ND", "SLICE", "SPLIT", "STRIDED_SLICE"}
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and rank in {4, 5}
                and op_type in {"GATHER", "GATHER_ND", "SLICE", "SPLIT", "STRIDED_SLICE"}
                and _is_degenerate_sequence_like_rank4_or_rank5_tensor(tensor)
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and op_type == "SOFTMAX"
                and (
                    _is_attention_like_softmax_op(model_ir, op)
                    or _is_transpose_sandwiched_last_axis_softmax_op(model_ir, op)
                )
            ):
                continue
            if (
                rank == 4
                and is_channel_last_logical_layout(layout)
                and _is_pytorch_channel_first_safe_rank4_island_op(model_ir, op)
            ):
                continue
            if layout == LOGICAL_LAYOUT_UNKNOWN or is_channel_last_logical_layout(layout):
                problems.append(
                    f"op_type={op_type} tensor={tensor_name} logical_layout={layout}"
                )
    if len(problems) > 0:
        raise ModelIRPyTorchExportError(
            "Channel-first normalization failed: semantic layout annotations are incomplete. "
            f"problems={sorted(set(problems))}"
        )


def _is_rank4_channel_last_dynamic_tensor(tensor: Optional[TensorIR]) -> bool:
    if tensor is None or isinstance(tensor.data, np.ndarray):
        return False
    return (
        len(list(tensor.shape)) == 4
        and is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout))
    )


def _is_pytorch_channel_first_safe_rank4_island_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    passthrough_op_types = {
        "ADD",
        "AVERAGE_POOL_2D",
        "CONCATENATION",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "DIV",
        "LEAKY_RELU",
        "LOGISTIC",
        "MAX_POOL_2D",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "RELU",
        "RELU6",
        "SUB",
        "TANH",
    }
    if op_type in passthrough_op_types:
        relevant_dynamic_tensors = [
            tensor
            for tensor_name in list(op.inputs) + list(op.outputs)
            for tensor in [model_ir.tensors.get(str(tensor_name), None)]
            if _is_rank4_channel_last_dynamic_tensor(tensor)
        ]
        return len(relevant_dynamic_tensors) > 0
    return False


def _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(
    model_ir: ModelIR,
) -> bool:
    public_boundary_names = [str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)]
    if len(public_boundary_names) == 0:
        return False
    for tensor_name in public_boundary_names:
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None:
            return False
        rank = len(list(tensor.shape))
        if rank not in {4, 5}:
            return False
        if not is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout)):
            return False
    if any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
        return False
    for op in model_ir.operators:
        if not _is_pytorch_channel_first_safe_rank4_island_op(model_ir, op):
            return False
    return True


def _shrink_preserved_channel_last_regions_for_pytorch(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> Set[str]:
    if len(preserve_channel_last_tensor_names) == 0:
        return set()
    if _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(model_ir):
        return {str(name) for name in preserve_channel_last_tensor_names}
    producers: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_idx)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    public_boundary_names = {str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)}
    shrunken_preserve_names: Set[str] = {
        str(name) for name in preserve_channel_last_tensor_names
    }
    for tensor_name in sorted(str(name) for name in preserve_channel_last_tensor_names):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if not _is_rank4_channel_last_dynamic_tensor(tensor):
            continue
        if str(tensor_name) in public_boundary_names:
            continue
        producer_idx = producers.get(str(tensor_name), None)
        if producer_idx is None:
            continue
        producer_op = model_ir.operators[int(producer_idx)]
        if not _is_pytorch_channel_first_safe_rank4_island_op(model_ir, producer_op):
            continue
        consumer_indices = consumers.get(str(tensor_name), [])
        if len(consumer_indices) == 0:
            continue
        if any(
            str(model_ir.operators[int(consumer_idx)].op_type) == "DEPTHWISE_CONV_2D"
            for consumer_idx in consumer_indices
        ):
            continue
        if any(
            not _is_pytorch_channel_first_safe_rank4_island_op(
                model_ir,
                model_ir.operators[int(consumer_idx)],
            )
            for consumer_idx in consumer_indices
        ):
            continue
        shrunken_preserve_names.discard(str(tensor_name))
    return shrunken_preserve_names


def _restore_non_preserved_channel_first_layouts(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    public_layout_bridge_tensor_names = {
        str(name)
        for name in list(model_ir.metadata.get("public_layout_bridge_tensor_names", []))
    }
    for tensor_name, tensor in model_ir.tensors.items():
        if str(tensor_name) in preserve_channel_last_tensor_names:
            continue
        if str(tensor_name) in public_layout_bridge_tensor_names:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        layout = normalize_logical_layout(tensor.logical_layout)
        if is_channel_last_logical_layout(layout):
            tensor.logical_layout = channel_first_logical_layout(rank)


def normalize_model_ir_for_pytorch_channel_first(model_ir: ModelIR) -> ModelIR:
    normalized = copy.deepcopy(model_ir)
    original_public_boundary_shapes: Dict[str, List[int]] = {}
    original_public_boundary_layouts: Dict[str, str] = {}
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        original_public_boundary_shapes[str(tensor_name)] = [
            int(v) for v in list(tensor.shape_signature or tensor.shape)
        ]
        original_public_boundary_layouts[str(tensor_name)] = normalize_logical_layout(
            tensor.logical_layout
        )
    infer_model_ir_logical_layouts(normalized)
    preserve_channel_last_tensor_names = _collect_feature_last_sequence_tensor_names(normalized)
    _apply_feature_last_sequence_layouts(normalized, preserve_channel_last_tensor_names)
    if len(preserve_channel_last_tensor_names) > 0:
        infer_model_ir_logical_layouts(normalized)
    preserve_channel_last_tensor_names = _shrink_preserved_channel_last_regions_for_pytorch(
        normalized,
        preserve_channel_last_tensor_names,
    )
    _apply_feature_last_sequence_layouts(normalized, preserve_channel_last_tensor_names)
    if len(preserve_channel_last_tensor_names) > 0:
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
    _rewrite_layout_sensitive_ops(normalized, original_layouts, preserve_channel_last_tensor_names)
    _propagate_pytorch_friendly_layouts(normalized)
    kernel_weight_tensor_names = _collect_kernel_weight_tensor_names(normalized)
    for tensor_name, tensor in normalized.tensors.items():
        if str(tensor_name) in kernel_weight_tensor_names:
            continue
        if str(tensor_name) in preserve_channel_last_tensor_names:
            continue
        _permute_tensor_to_channel_first_inplace(tensor)
    _synchronize_reshape_targets_with_output_tensors(normalized, preserve_channel_last_tensor_names)
    _rewrite_filter_tensors_for_pytorch(normalized)
    _remove_redundant_layout_transposes(normalized, original_layouts, preserve_channel_last_tensor_names)
    _propagate_pytorch_friendly_layouts(normalized)
    _apply_feature_last_sequence_layouts(normalized, preserve_channel_last_tensor_names)
    _restore_non_preserved_channel_first_layouts(normalized, preserve_channel_last_tensor_names)
    _rewrite_atan2_ones_like_to_atan(normalized)
    _repair_orphan_recurrent_step_tensors(normalized)
    public_layout_map = normalized.metadata.get("onnx_public_layout_map", None)
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
        normalized.metadata["onnx_public_layout_map"] = public_layout_map
    boundary_shape_map = normalized.metadata.get("onnx_boundary_shape_signature_map", None)
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
        normalized.metadata["onnx_boundary_shape_signature_map"] = boundary_shape_map
    preserve_public_channel_last_boundaries = _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(
        normalized
    )
    if isinstance(public_layout_map, dict):
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
                    or normalized_tensor_name in preserve_channel_last_tensor_names
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
                public_layout_map[normalized_output_name] = channel_last_logical_layout(output_rank)
        for op in normalized.operators:
            if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 1 or len(op.outputs) != 1:
                continue
            output_name = str(op.outputs[0])
            if output_name not in set(str(v) for v in normalized.outputs):
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
            if is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout)):
                public_layout_map[output_name] = channel_first_logical_layout(output_rank)
    _align_public_boundary_shapes_to_onnx_contract(normalized)
    normalized.metadata["assume_channel_last_layout_tensor_names"] = []
    _reject_residual_layout_transposes(normalized, preserve_channel_last_tensor_names)
    validate_channel_first_exportability(normalized, preserve_channel_last_tensor_names)
    return normalized


def _is_attention_like_softmax_op(model_ir: ModelIR, op: OperatorIR) -> bool:
    if str(op.op_type) != "SOFTMAX":
        return False
    reference_tensor: Optional[TensorIR] = None
    if len(op.inputs) > 0:
        reference_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    if reference_tensor is None and len(op.outputs) > 0:
        reference_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if reference_tensor is None:
        return False
    shape = [int(v) for v in list(reference_tensor.shape)]
    rank = len(shape)
    if rank < 3:
        return False
    axis = op.options.get("axis", None)
    resolved_axis = int(axis) if axis is not None else rank - 1
    if resolved_axis < 0:
        resolved_axis += rank
    if resolved_axis != rank - 1:
        return False
    if int(shape[-1]) <= 1:
        return False
    output_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
    if output_name != "":
        for consumer in model_ir.operators:
            if output_name not in [str(v) for v in consumer.inputs]:
                continue
            if str(consumer.op_type) == "BATCH_MATMUL":
                return True
    if rank == 3 and int(shape[-2]) == int(shape[-1]):
        return True
    if rank >= 4 and int(shape[-2]) == int(shape[-1]) and 0 < int(shape[-3]) <= 64:
        return True
    return False


def _is_transpose_sandwiched_last_axis_softmax_op(model_ir: ModelIR, op: OperatorIR) -> bool:
    if str(op.op_type) != "SOFTMAX" or len(op.inputs) < 1 or len(op.outputs) != 1:
        return False
    input_name = str(op.inputs[0])
    output_name = str(op.outputs[0])
    input_tensor = model_ir.tensors.get(input_name, None)
    output_tensor = model_ir.tensors.get(output_name, None)
    if input_tensor is None or output_tensor is None:
        return False
    rank = len(list(input_tensor.shape))
    if rank not in {3, 4, 5} or len(list(output_tensor.shape)) != rank:
        return False
    axis = int(op.options.get("axis", rank - 1))
    if axis < 0:
        axis += rank
    if axis != rank - 1:
        return False

    producer_op: Optional[OperatorIR] = None
    for candidate in model_ir.operators:
        if input_name in [str(v) for v in candidate.outputs]:
            producer_op = candidate
            break
    if producer_op is None or str(producer_op.op_type) != "TRANSPOSE" or len(producer_op.inputs) < 1:
        return False
    producer_perm = _read_transpose_perm(model_ir, producer_op)
    if (
        producer_perm is None
        or len(producer_perm) != rank
        or sorted(int(v) for v in producer_perm) != list(range(rank))
        or [int(v) for v in producer_perm] == list(range(rank))
    ):
        return False

    consumer_ops = [
        candidate
        for candidate in model_ir.operators
        if output_name in [str(v) for v in candidate.inputs]
    ]
    if len(consumer_ops) != 1:
        return False
    consumer_op = consumer_ops[0]
    if str(consumer_op.op_type) != "TRANSPOSE" or len(consumer_op.outputs) != 1:
        return False
    consumer_perm = _read_transpose_perm(model_ir, consumer_op)
    if consumer_perm is None or len(consumer_perm) != rank:
        return False
    inverse_perm = [0] * rank
    for new_axis, old_axis in enumerate(producer_perm):
        inverse_perm[int(old_axis)] = int(new_axis)
    if [int(v) for v in consumer_perm] != inverse_perm:
        return False

    source_tensor = model_ir.tensors.get(str(producer_op.inputs[0]), None)
    restored_tensor = model_ir.tensors.get(str(consumer_op.outputs[0]), None)
    if source_tensor is None or restored_tensor is None:
        return False
    source_layout = normalize_logical_layout(source_tensor.logical_layout)
    restored_layout = normalize_logical_layout(restored_tensor.logical_layout)
    if (
        source_layout == LOGICAL_LAYOUT_UNKNOWN
        or restored_layout == LOGICAL_LAYOUT_UNKNOWN
        or source_layout != restored_layout
    ):
        return False
    return True


def _is_layout_agnostic_native_model_ir(model_ir: ModelIR) -> bool:
    channel_sensitive_ops = {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "TRANSPOSE_CONV",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
        "MAX_POOL_2D",
        "AVERAGE_POOL_2D",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "NON_MAX_SUPPRESSION_V4",
    }
    op_types = _collect_model_op_types(model_ir)
    return len(op_types & channel_sensitive_ops) == 0


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


def _rewrite_recurrent_ops_for_native_export(model_ir: ModelIR) -> ModelIR:
    recurrent_op_types = {
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if not any(str(op.op_type) in recurrent_op_types for op in model_ir.operators):
        return copy.deepcopy(model_ir)
    if all(
        (
            str(op.op_type) == "UNIDIRECTIONAL_SEQUENCE_RNN"
            and _can_direct_codegen_sequence_rnn_op(model_ir, op)
        )
        or (
            str(op.op_type) in {"UNIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_LSTM"}
            and _can_direct_codegen_sequence_lstm_op(model_ir, op)
        )
        or str(op.op_type) not in recurrent_op_types
        for op in model_ir.operators
    ):
        return copy.deepcopy(model_ir)
    try:
        rewritten_model_ir, _ = rewrite_model_ir_unroll_recurrent_ops(
            model_ir=model_ir,
        )
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter could not rewrite recurrent sequence ops "
            "for native export."
        ) from ex
    return rewritten_model_ir


def _get_model_ir_subgraph_by_1based_index(
    model_ir: ModelIR,
    index: Any,
) -> Optional[ModelIR]:
    try:
        subgraph_index = int(index) - 1
    except Exception:
        return None
    if subgraph_index < 0 or subgraph_index >= len(model_ir.subgraphs):
        return None
    return model_ir.subgraphs[int(subgraph_index)]


def _constant_scalar_value(tensor: Optional[TensorIR]) -> Optional[Any]:
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    arr = np.asarray(tensor.data)
    if arr.size != 1:
        return None
    value = arr.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _reshape_alias_source_name(
    subgraph: ModelIR,
    tensor_name: str,
) -> Optional[str]:
    producer: Optional[OperatorIR] = None
    for op in subgraph.operators:
        if str(tensor_name) in {str(v) for v in op.outputs}:
            producer = op
            break
    if producer is None:
        return None
    if str(producer.op_type) != "RESHAPE" or len(producer.inputs) == 0:
        return None
    return str(producer.inputs[0])


def _is_canonical_imported_while_cond_subgraph(
    *,
    cond_subgraph: ModelIR,
    input_count: int,
    output_count: int,
    body_subgraph: ModelIR,
) -> bool:
    if (
        len(cond_subgraph.inputs) != input_count
        or len(cond_subgraph.outputs) != 1
        or len(body_subgraph.inputs) != input_count
        or len(body_subgraph.outputs) != output_count
        or len(cond_subgraph.operators) != 2
    ):
        return False
    cond_less_op = cond_subgraph.operators[0]
    cond_and_op = cond_subgraph.operators[1]
    return (
        str(cond_less_op.op_type) == "LESS"
        and len(cond_less_op.inputs) == 2
        and len(cond_less_op.outputs) == 1
        and str(cond_less_op.inputs[0]) == str(cond_subgraph.inputs[0])
        and str(cond_less_op.inputs[1]) == str(cond_subgraph.inputs[1])
        and str(cond_and_op.op_type) == "LOGICAL_AND"
        and len(cond_and_op.inputs) == 2
        and len(cond_and_op.outputs) == 1
        and str(cond_and_op.inputs[0]) == str(cond_subgraph.inputs[2])
        and str(cond_and_op.inputs[1]) == str(cond_less_op.outputs[0])
        and str(cond_and_op.outputs[0]) == str(cond_subgraph.outputs[0])
    )


def _match_static_unrollable_while_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> Optional[Dict[str, Any]]:
    if str(op.op_type) != "WHILE" or len(op.inputs) < 4 or len(op.outputs) != len(op.inputs):
        return None
    iter_value = _constant_scalar_value(model_ir.tensors.get(str(op.inputs[0]), None))
    trip_value = _constant_scalar_value(model_ir.tensors.get(str(op.inputs[1]), None))
    cond_value = _constant_scalar_value(model_ir.tensors.get(str(op.inputs[2]), None))
    if not isinstance(iter_value, (int, np.integer)):
        return None
    if not isinstance(trip_value, (int, np.integer)):
        return None
    if not isinstance(cond_value, (bool, np.bool_)):
        return None
    cond_subgraph = _get_model_ir_subgraph_by_1based_index(model_ir, op.options.get("condSubgraphIndex", 0))
    body_subgraph = _get_model_ir_subgraph_by_1based_index(model_ir, op.options.get("bodySubgraphIndex", 0))
    if cond_subgraph is None or body_subgraph is None:
        return None
    if not _is_canonical_imported_while_cond_subgraph(
        cond_subgraph=cond_subgraph,
        input_count=len(op.inputs),
        output_count=len(op.outputs),
        body_subgraph=body_subgraph,
    ):
        return None

    body_iter_in = str(body_subgraph.inputs[0])
    body_trip_in = str(body_subgraph.inputs[1])
    body_cond_in = str(body_subgraph.inputs[2])
    body_iter_out = str(body_subgraph.outputs[0])
    body_trip_out = str(body_subgraph.outputs[1])
    body_cond_out = str(body_subgraph.outputs[2])

    iter_out_producer: Optional[OperatorIR] = None
    for candidate in body_subgraph.operators:
        if body_iter_out in {str(v) for v in candidate.outputs}:
            iter_out_producer = candidate
            break
    if (
        iter_out_producer is None
        or str(iter_out_producer.op_type) != "ADD"
        or len(iter_out_producer.inputs) != 2
        or str(iter_out_producer.inputs[0]) != body_iter_in
    ):
        return None
    iter_plus_one_value = _constant_scalar_value(
        body_subgraph.tensors.get(str(iter_out_producer.inputs[1]), None)
    )
    if not isinstance(iter_plus_one_value, (int, np.integer)) or int(iter_plus_one_value) != 1:
        return None
    if _reshape_alias_source_name(body_subgraph, body_trip_out) != body_trip_in:
        return None
    if _reshape_alias_source_name(body_subgraph, body_cond_out) != body_cond_in:
        return None
    return {
        "iter_init": int(iter_value),
        "trip_count": int(trip_value),
        "cond_init": bool(cond_value),
        "body_subgraph": body_subgraph,
    }


def _match_counter_bounded_unrollable_while_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> Optional[Dict[str, Any]]:
    if str(op.op_type) != "WHILE" or len(op.inputs) < 5 or len(op.outputs) != len(op.inputs):
        return None
    cond_subgraph = _get_model_ir_subgraph_by_1based_index(
        model_ir,
        op.options.get("condSubgraphIndex", 0),
    )
    body_subgraph = _get_model_ir_subgraph_by_1based_index(
        model_ir,
        op.options.get("bodySubgraphIndex", 0),
    )
    if cond_subgraph is None or body_subgraph is None:
        return None
    if not _is_canonical_imported_while_cond_subgraph(
        cond_subgraph=cond_subgraph,
        input_count=len(op.inputs),
        output_count=len(op.outputs),
        body_subgraph=body_subgraph,
    ):
        return None
    for state_offset in range(4, len(op.inputs)):
        alias_source = _reshape_alias_source_name(body_subgraph, str(body_subgraph.outputs[state_offset]))
        if alias_source is not None:
            state_output_raw = str(alias_source)
        else:
            state_output_raw = f"{str(body_subgraph.outputs[state_offset])}_raw"
            if str(body_subgraph.outputs[state_offset]).endswith("_out"):
                state_output_raw = str(body_subgraph.outputs[state_offset]).replace("_out", "_out_raw")
            if state_output_raw not in body_subgraph.tensors:
                continue
        cond_producer: Optional[OperatorIR] = None
        for candidate in body_subgraph.operators:
            if str(body_subgraph.outputs[2]) in {str(v) for v in candidate.outputs}:
                cond_producer = candidate
                break
        if cond_producer is None:
            return None
        compare_output_name = str(body_subgraph.outputs[2])
        compare_op = cond_producer
        invert_compare = False
        if str(cond_producer.op_type) == "SELECT" and len(cond_producer.inputs) == 3:
            compare_output_name = str(cond_producer.inputs[0])
            true_name = str(cond_producer.inputs[1])
            false_name = str(cond_producer.inputs[2])
            true_value = _constant_scalar_value(body_subgraph.tensors.get(true_name, None))
            false_value = _constant_scalar_value(body_subgraph.tensors.get(false_name, None))
            if bool(true_value) is False and bool(false_value) is True:
                invert_compare = True
            else:
                return None
            compare_op = None
            for candidate in body_subgraph.operators:
                if compare_output_name in {str(v) for v in candidate.outputs}:
                    compare_op = candidate
                    break
            if compare_op is None:
                return None
        if str(compare_op.op_type) not in {"LESS", "GREATER_EQUAL"} or len(compare_op.inputs) != 2:
            return None
        lhs_name = str(compare_op.inputs[0])
        rhs_name = str(compare_op.inputs[1])
        lhs_cast: Optional[OperatorIR] = None
        rhs_cast: Optional[OperatorIR] = None
        for candidate in body_subgraph.operators:
            if lhs_name in {str(v) for v in candidate.outputs}:
                lhs_cast = candidate
            if rhs_name in {str(v) for v in candidate.outputs}:
                rhs_cast = candidate
        if lhs_cast is None or rhs_cast is None:
            return None
        if str(lhs_cast.op_type) != "CAST" or str(rhs_cast.op_type) != "CAST":
            return None
        lhs_source_name = str(lhs_cast.inputs[0]) if len(lhs_cast.inputs) >= 1 else ""
        rhs_source_name = str(rhs_cast.inputs[0]) if len(rhs_cast.inputs) >= 1 else ""
        if lhs_source_name != state_output_raw:
            continue
        threshold_value = _constant_scalar_value(body_subgraph.tensors.get(rhs_source_name, None))
        if not isinstance(threshold_value, (int, np.integer)):
            continue
        if str(compare_op.op_type) == "GREATER_EQUAL" and not invert_compare:
            continue
        return {
            "body_subgraph": body_subgraph,
            "max_iterations": max(1, int(threshold_value)),
        }
    return None


def _ensure_tensor_shape_literal(
    model_ir: ModelIR,
    *,
    base_name: str,
    shape: Sequence[int],
    used_names: Set[str],
) -> str:
    shape_tensor_name = _make_unique_identifier(f"{base_name}_shape", used_names)
    shape_values = [int(v) for v in list(shape)]
    model_ir.tensors[str(shape_tensor_name)] = TensorIR(
        name=str(shape_tensor_name),
        dtype="INT32",
        shape=[int(len(shape_values))],
        shape_signature=[int(len(shape_values))],
        data=np.asarray(shape_values, dtype=np.int32),
    )
    return str(shape_tensor_name)


def _rewrite_static_while_ops_for_native_export(model_ir: ModelIR) -> ModelIR:
    rewritten = copy.deepcopy(model_ir)
    used_names: Set[str] = set(str(name) for name in rewritten.tensors.keys())
    expanded_ops: List[OperatorIR] = []
    changed = False

    for op_index, op in enumerate(rewritten.operators):
        match = _match_static_unrollable_while_op(rewritten, op)
        if match is None:
            expanded_ops.append(op)
            continue
        changed = True
        body_subgraph = cast(ModelIR, match["body_subgraph"])
        iterations = (
            max(0, int(match["trip_count"]) - int(match["iter_init"]))
            if bool(match["cond_init"])
            else 0
        )

        constant_name_map: Dict[str, str] = {}
        for tensor_name, tensor in body_subgraph.tensors.items():
            if str(tensor_name) in {str(v) for v in body_subgraph.inputs}:
                continue
            if not isinstance(tensor.data, np.ndarray):
                continue
            mapped_name = str(tensor_name)
            if mapped_name in rewritten.tensors:
                mapped_name = _make_unique_identifier(mapped_name, used_names)
            else:
                used_names.add(mapped_name)
            cloned = _clone_tensor(tensor)
            cloned.name = str(mapped_name)
            rewritten.tensors[str(mapped_name)] = cloned
            constant_name_map[str(tensor_name)] = str(mapped_name)

        current_values = [str(v) for v in op.inputs]
        if iterations == 0:
            final_values = list(current_values)
            for output_index, output_name in enumerate(op.outputs):
                source_name = str(final_values[output_index])
                target_name = str(output_name)
                if source_name == target_name:
                    continue
                target_tensor = rewritten.tensors.get(target_name, None)
                if target_tensor is None:
                    continue
                shape_name = _ensure_tensor_shape_literal(
                    rewritten,
                    base_name=target_name,
                    shape=target_tensor.shape_signature or target_tensor.shape,
                    used_names=used_names,
                )
                expanded_ops.append(
                    OperatorIR(
                        op_type="RESHAPE",
                        inputs=[source_name, shape_name],
                        outputs=[target_name],
                        options={
                            "newShape": [
                                int(v) for v in list(target_tensor.shape_signature or target_tensor.shape)
                            ],
                            "allowZero": False,
                        },
                    )
                )
            continue

        body_output_index = {
            str(output_name): int(idx) for idx, output_name in enumerate(body_subgraph.outputs)
        }
        for iteration_index in range(iterations):
            local_map: Dict[str, str] = {
                str(body_subgraph.inputs[input_index]): str(current_values[input_index])
                for input_index in range(len(body_subgraph.inputs))
            }
            local_map.update(constant_name_map)
            is_last_iteration = int(iteration_index) == int(iterations) - 1
            for body_op in body_subgraph.operators:
                cloned_op = copy.deepcopy(body_op)
                cloned_op.inputs = [str(local_map.get(str(name), str(name))) for name in body_op.inputs]
                cloned_outputs: List[str] = []
                for output_name in body_op.outputs:
                    output_str = str(output_name)
                    if is_last_iteration and output_str in body_output_index:
                        mapped_name = str(op.outputs[int(body_output_index[output_str])])
                    else:
                        mapped_name = _make_unique_identifier(
                            f"{output_str}_unroll_{op_index}_{iteration_index}",
                            used_names,
                        )
                    if mapped_name not in rewritten.tensors and output_str in body_subgraph.tensors:
                        cloned_tensor = _clone_tensor(body_subgraph.tensors[output_str])
                        cloned_tensor.name = str(mapped_name)
                        rewritten.tensors[str(mapped_name)] = cloned_tensor
                    local_map[output_str] = str(mapped_name)
                    cloned_outputs.append(str(mapped_name))
                cloned_op.outputs = cloned_outputs
                expanded_ops.append(cloned_op)
            current_values = [str(local_map[str(name)]) for name in body_subgraph.outputs]

    if not changed:
        return rewritten
    rewritten.operators = expanded_ops
    return rewritten


def _rewrite_counter_bounded_while_ops_for_native_export(model_ir: ModelIR) -> ModelIR:
    rewritten = copy.deepcopy(model_ir)
    used_names: Set[str] = set(str(name) for name in rewritten.tensors.keys())
    expanded_ops: List[OperatorIR] = []
    changed = False

    for op_index, op in enumerate(rewritten.operators):
        match = _match_counter_bounded_unrollable_while_op(rewritten, op)
        if match is None:
            expanded_ops.append(op)
            continue
        changed = True
        body_subgraph = cast(ModelIR, match["body_subgraph"])
        max_iterations = int(match["max_iterations"])

        constant_name_map: Dict[str, str] = {}
        for tensor_name, tensor in body_subgraph.tensors.items():
            if str(tensor_name) in {str(v) for v in body_subgraph.inputs}:
                continue
            if not isinstance(tensor.data, np.ndarray):
                continue
            mapped_name = str(tensor_name)
            if mapped_name in rewritten.tensors:
                mapped_name = _make_unique_identifier(mapped_name, used_names)
            else:
                used_names.add(mapped_name)
            cloned = _clone_tensor(tensor)
            cloned.name = str(mapped_name)
            rewritten.tensors[str(mapped_name)] = cloned
            constant_name_map[str(tensor_name)] = str(mapped_name)

        current_values = [str(v) for v in op.inputs]
        for iteration_index in range(max_iterations):
            local_map: Dict[str, str] = {
                str(body_subgraph.inputs[input_index]): str(current_values[input_index])
                for input_index in range(len(body_subgraph.inputs))
            }
            local_map.update(constant_name_map)

            for body_op in body_subgraph.operators:
                cloned_op = copy.deepcopy(body_op)
                cloned_op.inputs = [str(local_map.get(str(name), str(name))) for name in body_op.inputs]
                cloned_outputs: List[str] = []
                for output_name in body_op.outputs:
                    output_str = str(output_name)
                    mapped_name = _make_unique_identifier(
                        f"{output_str}_while_mask_{op_index}_{iteration_index}",
                        used_names,
                    )
                    if mapped_name not in rewritten.tensors and output_str in body_subgraph.tensors:
                        cloned_tensor = _clone_tensor(body_subgraph.tensors[output_str])
                        cloned_tensor.name = str(mapped_name)
                        rewritten.tensors[str(mapped_name)] = cloned_tensor
                    local_map[output_str] = str(mapped_name)
                    cloned_outputs.append(str(mapped_name))
                cloned_op.outputs = cloned_outputs
                expanded_ops.append(cloned_op)

            next_values: List[str] = []
            current_cond_name = str(current_values[2])
            for output_index, output_name in enumerate(op.outputs):
                if output_index == 1:
                    next_values.append(str(local_map[str(body_subgraph.outputs[1])]))
                    continue
                if output_index == 2:
                    gated_name = (
                        str(output_name)
                        if iteration_index == max_iterations - 1
                        else _make_unique_identifier(f"{output_name}_while_masked", used_names)
                    )
                    if gated_name not in rewritten.tensors and str(output_name) in rewritten.tensors:
                        cloned_tensor = _clone_tensor(rewritten.tensors[str(output_name)])
                        cloned_tensor.name = str(gated_name)
                        rewritten.tensors[str(gated_name)] = cloned_tensor
                    expanded_ops.append(
                        OperatorIR(
                            op_type="LOGICAL_AND",
                            inputs=[current_cond_name, str(local_map[str(body_subgraph.outputs[2])])],
                            outputs=[str(gated_name)],
                            options={},
                        )
                    )
                    next_values.append(str(gated_name))
                    continue
                gated_name = (
                    str(output_name)
                    if iteration_index == max_iterations - 1
                    else _make_unique_identifier(f"{output_name}_while_masked", used_names)
                )
                if gated_name not in rewritten.tensors and str(output_name) in rewritten.tensors:
                    cloned_tensor = _clone_tensor(rewritten.tensors[str(output_name)])
                    cloned_tensor.name = str(gated_name)
                    rewritten.tensors[str(gated_name)] = cloned_tensor
                expanded_ops.append(
                    OperatorIR(
                        op_type="SELECT",
                        inputs=[
                            current_cond_name,
                            str(local_map[str(body_subgraph.outputs[output_index])]),
                            str(current_values[output_index]),
                        ],
                        outputs=[str(gated_name)],
                        options={},
                    )
                )
                next_values.append(str(gated_name))
            current_values = next_values

    if not changed:
        return rewritten
    rewritten.operators = expanded_ops
    return rewritten


def prepare_model_ir_for_native_pytorch(model_ir: ModelIR) -> ModelIR:
    original_boundary_shape_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(original_boundary_shape_map, dict):
        original_boundary_shape_map = {}
    original_public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(original_public_layout_map, dict):
        original_public_layout_map = {}
    original_public_boundary_shapes: Dict[str, List[int]] = {}
    original_public_boundary_layouts: Dict[str, str] = {}
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        boundary_shape = original_boundary_shape_map.get(str(tensor_name), tensor.shape_signature or tensor.shape)
        original_public_boundary_shapes[str(tensor_name)] = [int(v) for v in list(boundary_shape)]
        explicit_public_layout = original_public_layout_map.get(str(tensor_name), None)
        resolved_public_layout = normalize_logical_layout(
            explicit_public_layout if explicit_public_layout is not None else tensor.logical_layout
        )
        if (
            explicit_public_layout is None
            and is_channel_last_logical_layout(resolved_public_layout)
            and len(list(boundary_shape)) in {4, 5}
            and int(list(boundary_shape)[-1]) == 1
        ):
            resolved_public_layout = LOGICAL_LAYOUT_UNKNOWN
        original_public_boundary_layouts[str(tensor_name)] = resolved_public_layout
    rewritten_model_ir = _rewrite_recurrent_ops_for_native_export(
        _rewrite_counter_bounded_while_ops_for_native_export(
            _rewrite_static_while_ops_for_native_export(model_ir)
        )
    )
    try:
        prepared = normalize_model_ir_for_pytorch_channel_first(rewritten_model_ir)
    except ModelIRPyTorchExportError:
        if not _is_layout_agnostic_native_model_ir(rewritten_model_ir):
            raise
        prepared = copy.deepcopy(rewritten_model_ir)
        infer_model_ir_logical_layouts(prepared)
        prepared.metadata["assume_channel_last_layout_tensor_names"] = []
    prepared_public_layout_map = {
        str(name): str(layout)
        for name, layout in original_public_boundary_layouts.items()
        if layout != LOGICAL_LAYOUT_UNKNOWN
    }
    prepared.metadata["onnx_public_layout_map"] = prepared_public_layout_map
    prepared.metadata["onnx_boundary_shape_signature_map"] = {
        str(name): [int(v) for v in list(shape)]
        for name, shape in original_public_boundary_shapes.items()
        if str(name) in prepared_public_layout_map
        or str(name) in original_public_layout_map
    }
    _ensure_public_boundary_layout_bridges(
        model_ir=prepared,
        desired_public_shape_map=original_public_boundary_shapes,
        desired_public_layout_map=original_public_boundary_layouts,
    )
    _align_public_boundary_shapes_to_onnx_contract(prepared)
    return prepared


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
            if op_type not in SUPPORTED_TORCH_KERNEL_OP_TYPES
            and op_type not in _DIRECT_CODEGEN_SUPPORTED_OP_TYPES
            and op_type not in {"MODEL"}
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
    boundary_shape_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
    public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
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


def _is_runtime_wrapper_package_dir(package_dir: Path) -> bool:
    model_path = package_dir / "model.py"
    if not model_path.exists():
        return False
    try:
        model_source = model_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "load_generated_model_package" in model_source


_NUMPY_DTYPE_BY_TENSOR_DTYPE: Dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "INT8": np.dtype(np.int8),
    "INT16": np.dtype(np.int16),
    "INT32": np.dtype(np.int32),
    "INT64": np.dtype(np.int64),
    "UINT8": np.dtype(np.uint8),
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
    "FLOAT64": np.dtype(np.float64),
}


def _parse_torchscript_shape_hints(
    shape_hints: Optional[List[str]],
) -> Dict[str, List[int]]:
    if shape_hints is None:
        return {}
    parsed: Dict[str, List[int]] = {}
    for hint in shape_hints:
        parts = str(hint).split(":", maxsplit=1)
        if len(parts) != 2:
            continue
        input_name = str(parts[0]).strip()
        shape_str = str(parts[1]).strip()
        if input_name == "" or shape_str == "":
            continue
        try:
            parsed[input_name] = [int(v) for v in shape_str.split(",")]
        except Exception:
            continue
    return parsed


def _lookup_torchscript_shape_hint(
    *,
    input_name: str,
    shape_hints: Dict[str, List[int]],
    normalized_shape_hints: Dict[str, List[int]],
    normalize_name: Callable[[str], str],
) -> Optional[List[int]]:
    direct = shape_hints.get(str(input_name), None)
    if direct is not None:
        return [int(v) for v in list(direct)]
    normalized = normalized_shape_hints.get(normalize_name(str(input_name)), None)
    if normalized is not None:
        return [int(v) for v in list(normalized)]
    return None


def _resolve_torchscript_trace_shape(
    *,
    input_name: str,
    shape_values: Sequence[Any],
    shape_hint: Optional[Sequence[int]],
    export_label: str = "TorchScript export",
) -> Tuple[int, ...]:
    base_shape = [int(v) for v in list(shape_values)]
    if shape_hint is None:
        return _sanitize_torchscript_trace_shape(base_shape)
    hint_values = [int(v) for v in list(shape_hint)]
    if len(hint_values) != len(base_shape):
        raise ModelIRPyTorchExportError(
            f"{export_label} shape_hints rank mismatch. "
            f"input={input_name} expected_rank={len(base_shape)} actual_rank={len(hint_values)}"
        )
    resolved: List[int] = []
    for dim, hint_dim in zip(base_shape, hint_values):
        if int(dim) > 0:
            resolved.append(int(dim))
        elif int(hint_dim) > 0:
            resolved.append(int(hint_dim))
        else:
            raise ModelIRPyTorchExportError(
                f"{export_label} shape_hints must provide positive values for dynamic dimensions. "
                f"input={input_name} shape_hint={hint_values}"
            )
    return tuple(resolved)


def _load_torchscript_test_data_nhwc(
    test_data_nhwc_path: Optional[str],
) -> Optional[np.ndarray]:
    if not test_data_nhwc_path:
        return None
    if not os.path.exists(test_data_nhwc_path):
        raise FileNotFoundError(
            f"test_data_nhwc_path does not exist. path={test_data_nhwc_path}"
        )
    data = np.asarray(np.load(test_data_nhwc_path))
    if data.ndim != 4:
        raise ValueError(
            "test_data_nhwc_path must contain a 4D array [N,H,W,C]. "
            f"actual_shape={tuple(data.shape)}"
        )
    if int(data.shape[-1]) != 3:
        raise ValueError(
            "test_data_nhwc_path must have 3 channels in the last dim. "
            f"actual_shape={tuple(data.shape)}"
        )
    if int(data.shape[0]) <= 0:
        raise ValueError(
            "test_data_nhwc_path must include at least 1 sample. "
            f"actual_shape={tuple(data.shape)}"
        )
    return data


def _build_torchscript_image_input_from_nhwc(
    *,
    data: np.ndarray,
    expected_shape: Tuple[int, ...],
    np_dtype: np.dtype,
) -> np.ndarray:
    import tensorflow as tf

    if len(expected_shape) != 4:
        raise ValueError(
            "test_data_nhwc_path can only be used for rank-4 inputs. "
            f"expected_shape={expected_shape}"
        )

    expected_batch = int(expected_shape[0]) if int(expected_shape[0]) > 0 else int(data.shape[0])
    if data.shape[0] >= expected_batch:
        sample = np.asarray(data[:expected_batch])
    else:
        repeats = int(np.ceil(expected_batch / data.shape[0]))
        sample = np.concatenate([data] * repeats, axis=0)[:expected_batch]

    if int(expected_shape[1]) == 3:
        target_h = int(expected_shape[2])
        target_w = int(expected_shape[3])
        if int(sample.shape[1]) != target_h or int(sample.shape[2]) != target_w:
            sample = np.asarray(tf.image.resize(sample, [target_h, target_w]))
        sample = np.transpose(sample, [0, 3, 1, 2])
    elif int(expected_shape[3]) == 3:
        target_h = int(expected_shape[1])
        target_w = int(expected_shape[2])
        if int(sample.shape[1]) != target_h or int(sample.shape[2]) != target_w:
            sample = np.asarray(tf.image.resize(sample, [target_h, target_w]))
    else:
        raise ValueError(
            "test_data_nhwc_path can only be used for 3-channel image inputs. "
            f"expected_shape={expected_shape}"
        )
    return np.asarray(sample).astype(np_dtype, copy=False)


def _sanitize_torchscript_file_stem(name: str, *, fallback: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if sanitized == "":
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(fallback)).strip("_")
    if sanitized == "":
        sanitized = "model"
    return sanitized


def _sanitize_torchscript_trace_shape(values: Sequence[Any]) -> Tuple[int, ...]:
    sanitized: List[int] = []
    for value in list(values):
        dim = int(value)
        sanitized.append(dim if dim > 0 else 1)
    return tuple(sanitized)


def _can_autoresolve_batch_only_trace_shape(shape_values: Sequence[Any]) -> bool:
    values = [int(v) for v in list(shape_values)]
    if len(values) == 0:
        return False
    if int(values[0]) > 0:
        return False
    return all(int(v) > 0 for v in values[1:])


def _build_pytorch_export_example_inputs(
    *,
    package_dir: str,
    package_metadata: Dict[str, Any],
    custom_input_op_name_np_data_path: Optional[List[Any]],
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    export_label: str = "PyTorch export",
) -> Tuple[Tuple[Any, ...], Dict[str, List[int]], bool]:
    from onnx2tf.tflite_builder.accuracy_evaluator import (
        _generate_seeded_input,
        _extract_sample_from_custom,
        _fill_length_like_input,
        _load_custom_input_data,
        _normalize_tensor_name,
    )
    from onnx2tf.tflite_builder.pytorch_accuracy_evaluator import (
        _convert_inputs_for_package,
        _generate_string_input,
        _is_string_dtype,
    )

    input_names = [str(v) for v in list(package_metadata.get("inputs", []))]
    tensor_meta_map = package_metadata.get("tensors", {})
    if not isinstance(tensor_meta_map, dict):
        tensor_meta_map = {}
    custom_inputs = _load_custom_input_data(custom_input_op_name_np_data_path)
    test_data_nhwc = _load_torchscript_test_data_nhwc(test_data_nhwc_path)
    parsed_shape_hints = _parse_torchscript_shape_hints(shape_hints)
    normalized_shape_hints = {
        _normalize_tensor_name(str(input_name)): [int(v) for v in list(shape_value)]
        for input_name, shape_value in parsed_shape_hints.items()
    }
    normalized_custom_inputs = {
        _normalize_tensor_name(str(input_name)): value
        for input_name, value in custom_inputs.items()
    }

    def _lookup_custom_input(input_name: str) -> Optional[np.ndarray]:
        custom_value = custom_inputs.get(str(input_name), None)
        if custom_value is not None:
            return custom_value
        return normalized_custom_inputs.get(_normalize_tensor_name(str(input_name)), None)

    input_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]] = []
    dynamic_inputs_present = False
    missing_dynamic_hints: List[str] = []
    generated_inputs_np: Dict[str, np.ndarray] = {}
    for input_name in input_names:
        tensor_meta = tensor_meta_map.get(str(input_name), {})
        if not isinstance(tensor_meta, dict):
            raise ModelIRPyTorchExportError(
                f"PyTorch package metadata is missing tensor metadata for input '{input_name}'."
            )
        dtype_name = str(tensor_meta.get("dtype", "FLOAT32")).upper()
        if dtype_name not in _NUMPY_DTYPE_BY_TENSOR_DTYPE:
            raise ModelIRPyTorchExportError(
                f"Unsupported input dtype for {export_label}. input={input_name} dtype={dtype_name}"
            )
        shape_values = tensor_meta.get("shape_signature", tensor_meta.get("shape", []))
        if not isinstance(shape_values, list):
            shape_values = tensor_meta.get("shape", [])
        if not isinstance(shape_values, list):
            shape_values = []
        custom_input_value = _lookup_custom_input(str(input_name))
        shape_hint = _lookup_torchscript_shape_hint(
            input_name=str(input_name),
            shape_hints=parsed_shape_hints,
            normalized_shape_hints=normalized_shape_hints,
            normalize_name=_normalize_tensor_name,
        )
        has_dynamic_dim = any(int(v) <= 0 for v in list(shape_values))
        if has_dynamic_dim:
            dynamic_inputs_present = True
        trace_shape_values = _sanitize_torchscript_trace_shape(shape_values)
        dynamic_hint_resolved = False
        if custom_input_value is not None:
            trace_shape_values = tuple(
                int(v) for v in list(np.asarray(custom_input_value).shape)
            )
            dynamic_hint_resolved = True
        elif shape_hint is not None:
            trace_shape_values = _resolve_torchscript_trace_shape(
                input_name=str(input_name),
                shape_values=shape_values,
                shape_hint=shape_hint,
                export_label=export_label,
            )
            dynamic_hint_resolved = True
        elif (
            test_data_nhwc is not None
            and len(list(shape_values)) == 4
            and (
                int(shape_values[1]) in {3, -1, 0}
                or int(shape_values[3]) in {3, -1, 0}
            )
        ):
            trace_shape_values = _resolve_torchscript_trace_shape(
                input_name=str(input_name),
                shape_values=shape_values,
                shape_hint=[
                    int(test_data_nhwc.shape[0]),
                    int(test_data_nhwc.shape[-1]) if int(shape_values[1]) in {3, -1, 0} else int(test_data_nhwc.shape[1]),
                    int(test_data_nhwc.shape[1]) if int(shape_values[1]) in {3, -1, 0} else int(test_data_nhwc.shape[2]),
                    int(test_data_nhwc.shape[2]) if int(shape_values[1]) in {3, -1, 0} else int(test_data_nhwc.shape[-1]),
                ],
                export_label=export_label,
            )
            dynamic_hint_resolved = True
        elif _can_autoresolve_batch_only_trace_shape(shape_values):
            dynamic_hint_resolved = True
        input_specs.append(
            (
                str(input_name),
                _NUMPY_DTYPE_BY_TENSOR_DTYPE[dtype_name],
                trace_shape_values,
            )
        )
        if has_dynamic_dim and not dynamic_hint_resolved:
            missing_dynamic_hints.append(str(input_name))
            continue

        if custom_input_value is not None:
            generated_inputs_np[str(input_name)] = _extract_sample_from_custom(
                data=np.asarray(custom_input_value),
                sample_index=0,
                expected_shape=trace_shape_values,
                np_dtype=_NUMPY_DTYPE_BY_TENSOR_DTYPE[dtype_name],
            )
            continue
        if test_data_nhwc is not None and len(trace_shape_values) == 4:
            try:
                generated_inputs_np[str(input_name)] = _build_torchscript_image_input_from_nhwc(
                    data=test_data_nhwc,
                    expected_shape=trace_shape_values,
                    np_dtype=_NUMPY_DTYPE_BY_TENSOR_DTYPE[dtype_name],
                )
                continue
            except Exception as ex:
                if dynamic_hint_resolved and shape_hint is None and custom_input_value is None:
                    raise ModelIRPyTorchExportError(
                        f"{export_label} could not build an example input from test_data_nhwc_path. "
                        f"input={input_name} expected_shape={list(trace_shape_values)}"
                    ) from ex
    if len(missing_dynamic_hints) > 0:
        raise ModelIRPyTorchExportError(
            f"{export_label} requires concrete trace hints for all dynamic public inputs. "
            "Use --shape_hints as the recommended option, or provide "
            "--test_data_nhwc_path / custom_input_op_name_np_data_path when applicable. "
            f"package_dir={package_dir} missing_inputs={sorted(missing_dynamic_hints)}"
        )
    rng = np.random.default_rng(seed=0)
    example_inputs_np: Dict[str, np.ndarray] = {}
    for input_name, input_dtype, input_shape in input_specs:
        prebuilt = generated_inputs_np.get(str(input_name), None)
        if prebuilt is not None:
            example_inputs_np[str(input_name)] = np.asarray(prebuilt)
            continue
        if _is_string_dtype(np.dtype(input_dtype)):
            example_inputs_np[str(input_name)] = _generate_string_input(
                shape=input_shape,
                rng=rng,
            )
            continue
        if np.issubdtype(np.dtype(input_dtype), np.integer):
            canonical = _normalize_tensor_name(str(input_name))
            if "mask" in canonical.split("_"):
                example_inputs_np[str(input_name)] = np.ones(input_shape, dtype=input_dtype)
                continue
            if any(
                canonical.endswith(suffix)
                for suffix in ("length", "lengths", "len", "lens", "seq_len", "seq_lens")
            ):
                example_inputs_np[str(input_name)] = _fill_length_like_input(
                    input_name=str(input_name),
                    input_shape=input_shape,
                    input_dtype=np.dtype(input_dtype),
                    generated_inputs=example_inputs_np,
                )
                continue
        example_inputs_np[str(input_name)] = _generate_seeded_input(
            shape=input_shape,
            np_dtype=np.dtype(input_dtype),
            rng=rng,
        )
    converted_inputs = _convert_inputs_for_package(
        inputs=example_inputs_np,
        package_metadata=package_metadata,
    )
    example_input_shapes: Dict[str, List[int]] = {}
    ordered_inputs: List[Any] = []
    for input_name in input_names:
        input_value = converted_inputs.get(str(input_name), None)
        if input_value is None:
            raise ModelIRPyTorchExportError(
                f"{export_label} could not resolve an example input. input={input_name}"
            )
        if not hasattr(input_value, "shape"):
            raise ModelIRPyTorchExportError(
                f"{export_label} supports only tensor-like public inputs for native packages. "
                f"input={input_name} type={type(input_value).__name__}"
            )
        example_input_shapes[str(input_name)] = [int(v) for v in list(input_value.shape)]
        ordered_inputs.append(input_value)
    return tuple(ordered_inputs), example_input_shapes, bool(dynamic_inputs_present)


def _build_torchscript_example_inputs(
    *,
    package_dir: str,
    package_metadata: Dict[str, Any],
    custom_input_op_name_np_data_path: Optional[List[Any]],
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
) -> Tuple[Tuple[Any, ...], Dict[str, List[int]], bool]:
    return _build_pytorch_export_example_inputs(
        package_dir=package_dir,
        package_metadata=package_metadata,
        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
        shape_hints=shape_hints,
        test_data_nhwc_path=test_data_nhwc_path,
        export_label="TorchScript export",
    )


def _load_generated_package_export_metadata(
    *,
    package_dir: str,
    export_label: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    package_path = Path(package_dir)
    metadata_path = package_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"PyTorch package metadata is missing. path={metadata_path}"
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    package_init_path = package_path / "__init__.py"
    if not package_init_path.exists():
        raise FileNotFoundError(
            f"Generated PyTorch package is missing __init__.py. path={package_init_path}"
        )
    return package_path, metadata_path, metadata


def _write_generated_package_export_metadata(
    *,
    metadata_path: Path,
    metadata: Dict[str, Any],
    metadata_key: str,
    file_name: Optional[str],
    example_input_shapes: Dict[str, List[int]],
    dynamic_inputs_present: bool,
    error: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "file_name": file_name,
        "example_input_shapes": {
            str(name): [int(v) for v in list(shape)]
            for name, shape in example_input_shapes.items()
        },
        "dynamic_inputs_present": bool(dynamic_inputs_present),
    }
    if extra_fields is not None:
        payload.update(extra_fields)
    if error is not None:
        payload["error"] = str(error)
    metadata[str(metadata_key)] = payload
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _remove_generated_package_artifact_if_exists(artifact_path: Path) -> None:
    if not artifact_path.exists():
        return
    try:
        artifact_path.unlink()
    except Exception:
        pass


def _clear_onnx_graph_and_node_metadata_in_place(graph: onnx.GraphProto) -> None:
    del graph.metadata_props[:]
    for node in graph.node:
        del node.metadata_props[:]
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _clear_onnx_graph_and_node_metadata_in_place(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    _clear_onnx_graph_and_node_metadata_in_place(subgraph)


def _onnx_node_maps(
    graph: onnx.GraphProto,
) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, List[onnx.NodeProto]]]:
    producer_map: Dict[str, onnx.NodeProto] = {}
    consumer_map: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for output_name in node.output:
            producer_map[str(output_name)] = node
        for input_name in node.input:
            consumer_map.setdefault(str(input_name), []).append(node)
    return producer_map, consumer_map


def _onnx_node_attr(node: onnx.NodeProto, name: str) -> Optional[Any]:
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return None


def _onnx_set_node_attr(node: onnx.NodeProto, name: str, value: Any) -> None:
    new_attr = onnx.helper.make_attribute(str(name), value)
    for attr_index, attr in enumerate(node.attribute):
        if attr.name == name:
            node.attribute[attr_index].CopyFrom(new_attr)
            return
    node.attribute.append(new_attr)


def _onnx_replace_all_node_inputs(
    graph: onnx.GraphProto,
    *,
    old_name: str,
    new_name: str,
) -> None:
    if old_name == new_name:
        return
    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if str(input_name) == str(old_name):
                node.input[input_index] = str(new_name)
    for output in graph.output:
        if str(output.name) == str(old_name):
            output.name = str(new_name)


def _onnx_remove_nodes_by_name(
    graph: onnx.GraphProto,
    node_names: Sequence[str],
) -> None:
    remove_name_set = {str(name) for name in list(node_names)}
    if not remove_name_set:
        return
    kept_nodes = [node for node in graph.node if str(node.name) not in remove_name_set]
    del graph.node[:]
    graph.node.extend(kept_nodes)


def _onnx_get_initializer_index(graph: onnx.GraphProto, name: str) -> Optional[int]:
    for initializer_index, initializer in enumerate(graph.initializer):
        if str(initializer.name) == str(name):
            return int(initializer_index)
    return None


def _onnx_set_initializer_array(
    graph: onnx.GraphProto,
    *,
    name: str,
    array: np.ndarray,
) -> None:
    tensor = onnx.numpy_helper.from_array(np.asarray(array), name=str(name))
    initializer_index = _onnx_get_initializer_index(graph, str(name))
    if initializer_index is None:
        graph.initializer.append(tensor)
    else:
        graph.initializer[initializer_index].CopyFrom(tensor)


def _onnx_make_unique_initializer_name(graph: onnx.GraphProto, base_name: str) -> str:
    existing_names = {str(initializer.name) for initializer in graph.initializer}
    existing_names.update(str(node.output[0]) for node in graph.node if len(node.output) >= 1)
    candidate = str(base_name)
    suffix_index = 0
    while candidate in existing_names:
        suffix_index += 1
        candidate = f"{base_name}_{suffix_index}"
    return candidate


def _onnx_get_initializer_array(
    graph: onnx.GraphProto,
    name: str,
) -> Optional[np.ndarray]:
    initializer_index = _onnx_get_initializer_index(graph, str(name))
    if initializer_index is None:
        return None
    return onnx.numpy_helper.to_array(graph.initializer[initializer_index])


def _onnx_convert_pads_nhwc_to_nchw(pads: Sequence[int] | np.ndarray) -> Optional[np.ndarray]:
    pad_values = np.asarray(list(pads), dtype=np.int64).reshape(-1)
    if pad_values.size != 8:
        return None
    begin = pad_values[:4]
    end = pad_values[4:]
    reorder = [0, 3, 1, 2]
    return np.concatenate([begin[reorder], end[reorder]], axis=0).astype(np.int64)


def _onnx_fold_relu_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for transpose_node in list(graph.node):
        if str(transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        transpose_consumers = consumer_map.get(str(transpose_node.output[0]), [])
        if len(transpose_consumers) != 1:
            continue
        relu_node = transpose_consumers[0]
        if str(relu_node.op_type) != "Relu":
            continue
        relu_consumers = consumer_map.get(str(relu_node.output[0]), [])
        if len(relu_consumers) != 1:
            continue
        inverse_node = relu_consumers[0]
        if str(inverse_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        relu_node.input[0] = str(transpose_node.input[0])
        relu_node.output[0] = str(inverse_node.output[0])
        remove_node_names.extend([str(transpose_node.name), str(inverse_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_reducesum_sigmoid_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for transpose_node in list(graph.node):
        if str(transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        transpose_consumers = consumer_map.get(str(transpose_node.output[0]), [])
        if len(transpose_consumers) != 1:
            continue
        reduce_node = transpose_consumers[0]
        if str(reduce_node.op_type) != "ReduceSum":
            continue
        axes_name = str(reduce_node.input[1]) if len(reduce_node.input) >= 2 else ""
        axes_array = _onnx_get_initializer_array(graph, axes_name)
        if axes_array is None or [int(v) for v in axes_array.reshape(-1)] != [3]:
            continue
        if int(_onnx_node_attr(reduce_node, "keepdims") or 0) != 1:
            continue
        reduce_consumers = consumer_map.get(str(reduce_node.output[0]), [])
        if len(reduce_consumers) != 1:
            continue
        sigmoid_node = reduce_consumers[0]
        if str(sigmoid_node.op_type) != "Sigmoid":
            continue
        sigmoid_consumers = consumer_map.get(str(sigmoid_node.output[0]), [])
        if len(sigmoid_consumers) != 1:
            continue
        inverse_node = sigmoid_consumers[0]
        if str(inverse_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        new_axes_name = _onnx_make_unique_initializer_name(graph, f"{axes_name}_nchw")
        _onnx_set_initializer_array(graph, name=new_axes_name, array=np.asarray([1], dtype=np.int64))
        reduce_node.input[0] = str(transpose_node.input[0])
        reduce_node.input[1] = str(new_axes_name)
        sigmoid_node.output[0] = str(inverse_node.output[0])
        remove_node_names.extend([str(transpose_node.name), str(inverse_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_inverse_transpose_pairs_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for first_node in list(graph.node):
        if str(first_node.op_type) != "Transpose":
            continue
        first_perm = [int(v) for v in list(_onnx_node_attr(first_node, "perm") or [])]
        if not first_perm:
            continue
        first_consumers = consumer_map.get(str(first_node.output[0]), [])
        if len(first_consumers) != 1:
            continue
        second_node = first_consumers[0]
        if str(second_node.op_type) != "Transpose":
            continue
        second_perm = [int(v) for v in list(_onnx_node_attr(second_node, "perm") or [])]
        inverse_perm = [0] * len(first_perm)
        for perm_index, perm_value in enumerate(first_perm):
            inverse_perm[int(perm_value)] = int(perm_index)
        if second_perm != inverse_perm:
            continue
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(second_node.output[0]),
            new_name=str(first_node.input[0]),
        )
        remove_node_names.extend([str(first_node.name), str(second_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_remove_passthrough_identity_nodes_in_place(graph: onnx.GraphProto) -> None:
    remove_node_names: List[str] = []
    for node in list(graph.node):
        if str(node.op_type) != "Identity":
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            continue
        input_name = str(node.input[0])
        output_name = str(node.output[0])
        if not input_name or not output_name or input_name == output_name:
            continue
        _onnx_replace_all_node_inputs(
            graph,
            old_name=output_name,
            new_name=input_name,
        )
        remove_node_names.append(str(node.name))
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_optimize_pidnet_spp_transpose_bridges_in_place(graph: onnx.GraphProto) -> None:
    node_by_name = {str(node.name): node for node in graph.node}
    required_node_names = {
        "node_permute_27",
        "node_permute_28",
        "node_pad_38",
        "node_pad_39",
        "node_pad_40",
        "node_permute_33",
        "node_permute_36",
        "node_avg_pool2d_1",
        "node_avg_pool2d_2",
        "node_mean",
        "node_mul_6",
        "node_mul_7",
        "node_mul_11",
        "node_permute_41",
        "node_add_28",
    }
    if any(node_name not in node_by_name for node_name in required_node_names):
        return

    node_permute_27 = node_by_name["node_permute_27"]
    node_permute_28 = node_by_name["node_permute_28"]
    node_pad_38 = node_by_name["node_pad_38"]
    node_pad_39 = node_by_name["node_pad_39"]
    node_pad_40 = node_by_name["node_pad_40"]
    node_permute_33 = node_by_name["node_permute_33"]
    node_permute_36 = node_by_name["node_permute_36"]
    node_avg_pool2d_1 = node_by_name["node_avg_pool2d_1"]
    node_avg_pool2d_2 = node_by_name["node_avg_pool2d_2"]
    node_mean = node_by_name["node_mean"]
    node_mul_6 = node_by_name["node_mul_6"]
    node_mul_7 = node_by_name["node_mul_7"]
    node_mul_11 = node_by_name["node_mul_11"]
    node_permute_41 = node_by_name["node_permute_41"]
    node_add_28 = node_by_name["node_add_28"]

    if list(_onnx_node_attr(node_permute_27, "perm") or []) != [0, 2, 3, 1]:
        return
    if list(_onnx_node_attr(node_permute_28, "perm") or []) != [0, 3, 1, 2]:
        return
    if list(_onnx_node_attr(node_permute_33, "perm") or []) != [0, 3, 1, 2]:
        return
    if list(_onnx_node_attr(node_permute_36, "perm") or []) != [0, 3, 1, 2]:
        return
    if list(_onnx_node_attr(node_permute_41, "perm") or []) != [0, 3, 1, 2]:
        return

    base_input_name = str(node_permute_27.input[0])
    node_mul_6.input[0] = base_input_name
    node_pad_38.input[0] = base_input_name
    node_mul_7.input[0] = base_input_name
    node_pad_39.input[0] = base_input_name
    node_pad_40.input[0] = base_input_name
    node_mean.input[0] = base_input_name
    node_avg_pool2d_1.input[0] = str(node_pad_39.output[0])
    node_avg_pool2d_2.input[0] = str(node_pad_40.output[0])
    node_add_28.input[0] = str(node_mul_11.output[0])

    mean_axes_name = str(node_mean.input[1]) if len(node_mean.input) >= 2 else ""
    if mean_axes_name:
        new_mean_axes_name = _onnx_make_unique_initializer_name(graph, f"{mean_axes_name}_nchw")
        _onnx_set_initializer_array(graph, name=new_mean_axes_name, array=np.asarray([2, 3], dtype=np.int64))
        node_mean.input[1] = str(new_mean_axes_name)

    for pad_node in [node_pad_39, node_pad_40]:
        pad_name = str(pad_node.input[1]) if len(pad_node.input) >= 2 else ""
        pad_values = _onnx_get_initializer_array(graph, pad_name)
        nchw_pad_values = (
            _onnx_convert_pads_nhwc_to_nchw(pad_values)
            if pad_values is not None
            else None
        )
        if pad_name and nchw_pad_values is not None:
            new_pad_name = _onnx_make_unique_initializer_name(graph, f"{pad_name}_nchw")
            _onnx_set_initializer_array(graph, name=new_pad_name, array=nchw_pad_values)
            pad_node.input[1] = str(new_pad_name)

    mul_const_name = str(node_mul_11.input[1]) if len(node_mul_11.input) >= 2 else ""
    mul_const_array = _onnx_get_initializer_array(graph, mul_const_name)
    if mul_const_name and mul_const_array is not None and len(mul_const_array.shape) == 4:
        _onnx_set_initializer_array(
            graph,
            name=mul_const_name,
            array=np.transpose(mul_const_array, (0, 3, 1, 2)),
        )

    _onnx_remove_nodes_by_name(
        graph,
        [
            str(node_permute_27.name),
            str(node_permute_28.name),
            str(node_permute_33.name),
            str(node_permute_36.name),
            str(node_permute_41.name),
        ],
    )


def _onnx_optimize_pphumanseg_add_resize_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        relu_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        relu_node = producer_map.get(relu_node_name)
        if relu_node is None or str(relu_node.op_type) != "Relu":
            continue
        if len(consumer_map.get(str(relu_node.output[0]), [])) != 1:
            continue
        add_node_name = str(relu_node.input[0]) if relu_node.input else ""
        add_node = producer_map.get(add_node_name)
        if add_node is None or str(add_node.op_type) != "Add" or len(add_node.input) != 2:
            continue
        if len(consumer_map.get(str(add_node.output[0]), [])) != 1:
            continue
        add_input_producers: List[onnx.NodeProto] = []
        for add_input_name in add_node.input:
            producer = producer_map.get(str(add_input_name))
            if producer is None:
                add_input_producers = []
                break
            add_input_producers.append(producer)
        if len(add_input_producers) != 2:
            continue
        if any(str(node.op_type) != "Transpose" for node in add_input_producers):
            continue
        if any(list(_onnx_node_attr(node, "perm") or []) != [0, 2, 3, 1] for node in add_input_producers):
            continue
        if any(len(consumer_map.get(str(node.output[0]), [])) != 1 for node in add_input_producers):
            continue
        resize_nodes = consumer_map.get(str(inverse_transpose_node.output[0]), [])
        if len(resize_nodes) != 1:
            continue
        resize_node = resize_nodes[0]
        if str(resize_node.op_type) != "Resize" or not resize_node.input:
            continue
        if str(resize_node.input[0]) != str(inverse_transpose_node.output[0]):
            continue
        if len(consumer_map.get(str(resize_node.output[0]), [])) != 1:
            continue
        trailing_nodes = consumer_map.get(str(resize_node.output[0]), [])
        if len(trailing_nodes) != 1:
            continue
        trailing_transpose_node = trailing_nodes[0]
        if str(trailing_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(trailing_transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue

        add_node.input[0] = str(add_input_producers[0].input[0])
        add_node.input[1] = str(add_input_producers[1].input[0])
        resize_node.input[0] = str(relu_node.output[0])
        remove_node_names.extend(
            [
                str(add_input_producers[0].name),
                str(add_input_producers[1].name),
                str(inverse_transpose_node.name),
            ]
        )
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_concat_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        concat_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        concat_node = producer_map.get(concat_node_name)
        if concat_node is None or str(concat_node.op_type) != "Concat":
            continue
        if int(_onnx_node_attr(concat_node, "axis") or -1) != 3:
            continue
        if len(consumer_map.get(str(concat_node.output[0]), [])) != 1:
            continue
        input_transpose_nodes: List[onnx.NodeProto] = []
        for input_name in concat_node.input:
            transpose_node = producer_map.get(str(input_name))
            if transpose_node is None:
                input_transpose_nodes = []
                break
            input_transpose_nodes.append(transpose_node)
        if not input_transpose_nodes:
            continue
        if any(str(node.op_type) != "Transpose" for node in input_transpose_nodes):
            continue
        if any(list(_onnx_node_attr(node, "perm") or []) != [0, 2, 3, 1] for node in input_transpose_nodes):
            continue
        if any(len(consumer_map.get(str(node.output[0]), [])) != 1 for node in input_transpose_nodes):
            continue

        for input_index, transpose_node in enumerate(input_transpose_nodes):
            concat_node.input[input_index] = str(transpose_node.input[0])
        _onnx_set_node_attr(concat_node, "axis", 1)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_transpose_node.output[0]),
            new_name=str(concat_node.output[0]),
        )
        remove_node_names.extend(
            [str(node.name) for node in input_transpose_nodes] + [str(inverse_transpose_node.name)]
        )
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_softmax_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        softmax_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        softmax_node = producer_map.get(softmax_node_name)
        if softmax_node is None or str(softmax_node.op_type) != "Softmax":
            continue
        if int(_onnx_node_attr(softmax_node, "axis") or -1) != 3:
            continue
        if len(consumer_map.get(str(softmax_node.output[0]), [])) != 1:
            continue
        if not softmax_node.input:
            continue
        input_transpose_node = producer_map.get(str(softmax_node.input[0]))
        if input_transpose_node is None or str(input_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(input_transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        if len(consumer_map.get(str(input_transpose_node.output[0]), [])) != 1:
            continue

        softmax_node.input[0] = str(input_transpose_node.input[0])
        _onnx_set_node_attr(softmax_node, "axis", 1)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_transpose_node.output[0]),
            new_name=str(softmax_node.output[0]),
        )
        remove_node_names.extend(
            [
                str(input_transpose_node.name),
                str(inverse_transpose_node.name),
            ]
        )
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _optimize_dynamo_exported_onnx_in_place(model: onnx.ModelProto) -> None:
    _onnx_remove_passthrough_identity_nodes_in_place(model.graph)
    _onnx_fold_relu_layout_bridges_in_place(model.graph)
    _onnx_fold_reducesum_sigmoid_layout_bridges_in_place(model.graph)
    _onnx_fold_inverse_transpose_pairs_in_place(model.graph)
    _onnx_optimize_pidnet_spp_transpose_bridges_in_place(model.graph)
    _onnx_optimize_pphumanseg_add_resize_layout_bridges_in_place(model.graph)
    _onnx_fold_concat_layout_bridges_in_place(model.graph)
    _onnx_fold_softmax_layout_bridges_in_place(model.graph)


def _onnx_model_uses_external_data(model: onnx.ModelProto) -> bool:
    return any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
    )


def _inspect_onnx_uses_external_data(onnx_path: Path) -> bool:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return _onnx_model_uses_external_data(model)


def _sanitize_dynamo_exported_onnx_metadata(onnx_path: Path) -> None:
    external_data_sidecar_path = onnx_path.with_name(f"{onnx_path.name}.data")
    original_uses_external_data = _inspect_onnx_uses_external_data(onnx_path)
    model = onnx.load(str(onnx_path))
    del model.metadata_props[:]
    _clear_onnx_graph_and_node_metadata_in_place(model.graph)
    _optimize_dynamo_exported_onnx_in_place(model)
    onnx.checker.check_model(model)
    if original_uses_external_data:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_sidecar_path.name,
            size_threshold=0,
        )
    else:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=False,
        )
    if not _inspect_onnx_uses_external_data(onnx_path) and external_data_sidecar_path.exists():
        external_data_sidecar_path.unlink()


def _metadata_has_dynamic_public_inputs(metadata: Dict[str, Any]) -> bool:
    tensor_meta_map = metadata.get("tensors", {})
    if not isinstance(tensor_meta_map, dict):
        return False
    for input_name in [str(v) for v in list(metadata.get("inputs", []))]:
        tensor_meta = tensor_meta_map.get(str(input_name), {})
        if not isinstance(tensor_meta, dict):
            continue
        shape_values = tensor_meta.get("shape_signature", tensor_meta.get("shape", []))
        if not isinstance(shape_values, list):
            shape_values = tensor_meta.get("shape", [])
        if not isinstance(shape_values, list):
            continue
        if any(int(v) <= 0 for v in list(shape_values)):
            return True
    return False


def _generated_package_non_native_skip_reason(package_path: Path) -> Optional[str]:
    metadata_path = package_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    execution_backend = str(metadata.get("execution_backend", "")).strip().lower()
    if execution_backend == "" and _is_runtime_wrapper_package_dir(package_path):
        execution_backend = "runtime_wrapper"
    if execution_backend not in {"", "native"}:
        return (
            "artifact export is skipped for generated packages with non-native execution "
            f"backend. execution_backend={execution_backend or 'native'}"
        )
    return None


def _generated_package_torch_export_skip_reason(package_path: Path) -> Optional[str]:
    non_native_skip_reason = _generated_package_non_native_skip_reason(package_path)
    if non_native_skip_reason is not None:
        return non_native_skip_reason
    model_path = package_path / "model.py"
    if not model_path.exists():
        return None
    try:
        model_source = model_path.read_text(encoding="utf-8")
    except Exception:
        return None
    if re.search(
        r"def _run_nms_\d+\(self, boxes: torch\.Tensor, scores: torch\.Tensor, max_output_size: torch\.Tensor",
        model_source,
    ):
        return (
            "torch.export-based artifacts are skipped for generated packages "
            "with data-dependent NON_MAX_SUPPRESSION_V4 parameters."
        )
    if (
        "_apply_non_max_suppression_v4(" in model_source
        and (
            "torch.as_tensor(min(2147483647, (_shape_list(" in model_source
            or "torch.as_tensor(min(2147483647, (_tensor_shape_list(" in model_source
            or (
                "torch.as_tensor(min(2147483647, " in model_source
                and ".shape[" in model_source
            )
        )
    ):
        return (
            "torch.export-based artifacts are skipped for generated packages "
            "with data-dependent NON_MAX_SUPPRESSION_V4 parameters."
        )

    if re.search(
        r"selected_indices_nms_valid_indices_c\d+\s*=\s*torch\.arange\(\s*start=0,\s*"
        r"end=selected_indices_nms_valid_count_scalar_c\d+\.reshape\(-1\)\[0\]\.item\(\)",
        model_source,
    ):
        return (
            "torch.export-based artifacts are skipped for generated packages "
            "with data-dependent NON_MAX_SUPPRESSION output-shape post-processing."
        )
    return None


def _run_generated_package_export_child(
    *,
    example_inputs: Tuple[Any, ...],
    child_script: str,
    package_path: Path,
    artifact_path: Path,
    child_payload: Dict[str, Any],
    child_args: Optional[List[str]] = None,
    temp_prefix: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "PyTorch export child execution requires `torch` to be installed."
        ) from ex

    if child_args is None:
        child_args = []
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        serialized_inputs_path = os.path.join(temp_dir, "example_inputs.pt")
        payload = dict(child_payload)
        payload["inputs"] = tuple(example_inputs)
        torch.save(payload, serialized_inputs_path)
        child_result = subprocess.run(
            [
                sys.executable,
                "-c",
                child_script,
                str(package_path),
                str(serialized_inputs_path),
                str(artifact_path),
                *[str(v) for v in child_args],
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if child_result.returncode == 0:
        try:
            return json.loads(child_result.stdout.strip() or "{}"), ""
        except json.JSONDecodeError:
            return {}, ""
    stderr_text = child_result.stderr.strip()
    stdout_text = child_result.stdout.strip()
    return None, (
        f"returncode={child_result.returncode} "
        f"stdout={stdout_text} stderr={stderr_text}"
    )


def export_torchscript_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Optional[str]:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "TorchScript export requires `torch` to be installed."
        ) from ex

    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="TorchScript export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
            extra_fields={
                "trace_mode": None,
            },
        )
        return None
    skip_reason = _generated_package_non_native_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "trace_mode": None,
                "skipped_reason": skip_reason,
            },
        )
        return None
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = _build_pytorch_export_example_inputs(
            package_dir=package_dir,
            package_metadata=metadata,
            custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
            shape_hints=shape_hints,
            test_data_nhwc_path=test_data_nhwc_path,
            export_label="TorchScript export",
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
            extra_fields={
                "trace_mode": None,
            },
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    torchscript_file_name = f"{file_stem}_jit.pt"
    torchscript_path = package_path / torchscript_file_name
    child_script = """
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
torchscript_path = Path(sys.argv[3])
mode = str(sys.argv[4]).strip().lower()

module_name = (
    "_onnx2tf_generated_torchscript_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
runtime_module = importlib.import_module(f"{module_name}.runtime")
setattr(runtime_module, "_ONNX2TF_DISABLE_SYMBOLIC_SHAPE_TENSORS", True)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
with torch.no_grad():
    if mode == "trace":
        artifact = torch.jit.trace(model, example_inputs, check_trace=False)
    elif mode == "script":
        artifact = torch.jit.script(model)
    else:
        raise RuntimeError(f"Unsupported torchscript export mode: {mode}")
    torch.jit.save(artifact, str(torchscript_path))
print(json.dumps({"trace_mode": mode}))
"""
    trace_mode = ""
    last_error_message = ""
    for candidate_mode in ("trace", "script"):
        child_payload, last_error_message = _run_generated_package_export_child(
            example_inputs=example_inputs,
            child_script=child_script,
            package_path=package_path,
            artifact_path=torchscript_path,
            child_payload={},
            child_args=[candidate_mode],
            temp_prefix="onnx2tf_torchscript_",
        )
        if child_payload is not None:
            trace_mode = str(child_payload.get("trace_mode", candidate_mode))
            break
        if last_error_message != "":
            last_error_message = f"mode={candidate_mode} {last_error_message}"
    if trace_mode == "":
        _remove_generated_package_artifact_if_exists(torchscript_path)
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message,
            extra_fields={
                "trace_mode": None,
            },
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "TorchScript export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        return None
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="torchscript",
        file_name=str(torchscript_file_name),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
        extra_fields={
            "trace_mode": trace_mode,
        },
    )
    return str(torchscript_path)


def export_dynamo_onnx_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Optional[str]:
    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="Dynamo ONNX export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        return None
    skip_reason = _generated_package_torch_export_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "skipped_reason": skip_reason,
            },
        )
        return None
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = _build_pytorch_export_example_inputs(
            package_dir=package_dir,
            package_metadata=metadata,
            custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
            shape_hints=shape_hints,
            test_data_nhwc_path=test_data_nhwc_path,
            export_label="Dynamo ONNX export",
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    dynamo_onnx_file_name = f"{file_stem}_dynamo.onnx"
    dynamo_onnx_path = package_path / dynamo_onnx_file_name
    child_script = """
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
dynamo_onnx_path = Path(sys.argv[3])

module_name = (
    "_onnx2tf_generated_dynamo_onnx_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
runtime_module = importlib.import_module(f"{module_name}.runtime")
setattr(runtime_module, "_ONNX2TF_DISABLE_SYMBOLIC_SHAPE_TENSORS", True)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
input_names = [str(v) for v in list(payload.get("input_names", []))]
output_names = [str(v) for v in list(payload.get("output_names", []))]
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
setattr(model, "_onnx2tf_torch_export_mode", True)
with torch.no_grad():
    torch.onnx.export(
        model,
        example_inputs,
        str(dynamo_onnx_path),
        dynamo=True,
        input_names=input_names,
        output_names=output_names,
    )
print(json.dumps({"file_name": dynamo_onnx_path.name}))
"""
    child_payload, last_error_message = _run_generated_package_export_child(
        example_inputs=example_inputs,
        child_script=child_script,
        package_path=package_path,
        artifact_path=dynamo_onnx_path,
        child_payload={
            "input_names": [str(v) for v in list(metadata.get("inputs", []))],
            "output_names": [str(v) for v in list(metadata.get("outputs", []))],
        },
        temp_prefix="onnx2tf_dynamo_onnx_",
    )
    if child_payload is None or not dynamo_onnx_path.exists():
        _remove_generated_package_artifact_if_exists(dynamo_onnx_path)
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message or "dynamo=True ONNX export did not produce an artifact.",
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "Dynamo ONNX export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        return None
    _sanitize_dynamo_exported_onnx_metadata(dynamo_onnx_path)
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="dynamo_onnx",
        file_name=str(child_payload.get("file_name", dynamo_onnx_file_name)),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
    )
    return str(dynamo_onnx_path)


def export_exported_program_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Optional[str]:
    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="ExportedProgram export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        return None
    skip_reason = _generated_package_torch_export_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "skipped_reason": skip_reason,
            },
        )
        return None
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = _build_pytorch_export_example_inputs(
            package_dir=package_dir,
            package_metadata=metadata,
            custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
            shape_hints=shape_hints,
            test_data_nhwc_path=test_data_nhwc_path,
            export_label="ExportedProgram export",
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    exported_program_file_name = f"{file_stem}_ep.pt2"
    exported_program_path = package_path / exported_program_file_name
    child_script = """
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
exported_program_path = Path(sys.argv[3])

module_name = (
    "_onnx2tf_generated_exported_program_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
setattr(model, "_onnx2tf_torch_export_mode", True)

def _prune_alias_nodes(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    graph_signature = getattr(exported_program, "graph_signature", None)
    user_output_names = set()
    if graph_signature is not None:
        try:
            user_output_names = {str(name) for name in list(graph_signature.user_outputs)}
        except Exception:
            user_output_names = set()
    changed = False
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and str(node.target) == "aten.alias.default"
            and len(node.args) >= 1
            and isinstance(node.args[0], torch.fx.Node)
            and str(node.name) not in user_output_names
        ):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

def _fold_inverse_permute_round_trips(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    changed = False

    def _normalize_perm(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _inverse_perm(perm):
        inverse = [0] * len(perm)
        for idx, value in enumerate(perm):
            inverse[int(value)] = int(idx)
        return inverse

    def _match_permute_chain_source(node, perm):
        if not isinstance(node, torch.fx.Node):
            return None
        if (
            node.op == "call_function"
            and str(node.target) == "aten.permute.default"
            and len(node.args) >= 2
            and _normalize_perm(node.args[1]) == perm
            and isinstance(node.args[0], torch.fx.Node)
        ):
            return node.args[0]
        if (
            node.op == "call_function"
            and str(node.target) == "aten.contiguous.default"
            and len(node.args) >= 1
            and isinstance(node.args[0], torch.fx.Node)
        ):
            input_node = node.args[0]
            if (
                input_node.op == "call_function"
                and str(input_node.target) == "aten.permute.default"
                and len(input_node.args) >= 2
                and _normalize_perm(input_node.args[1]) == perm
                and isinstance(input_node.args[0], torch.fx.Node)
            ):
                return input_node.args[0]
        return None

    def _match_binary_input_source(node, perm):
        return _match_permute_chain_source(node, perm)

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        perm = _normalize_perm(node.args[1])
        if perm is None:
            continue
        root_replacement = source
        root_users = list(node.users)
        if len(root_users) == 1 and (
            root_users[0].op == "call_function"
            and str(root_users[0].target) == "aten.contiguous.default"
        ):
            branch_input = root_users[0]
        else:
            branch_input = node
        branch_users = list(branch_input.users)
        if len(branch_users) == 1:
            inverse_node = branch_users[0]
            if (
                inverse_node.op == "call_function"
                and str(inverse_node.target) == "aten.permute.default"
                and len(inverse_node.args) >= 2
            ):
                inverse_perm = _normalize_perm(inverse_node.args[1])
                if inverse_perm is not None and inverse_perm == _inverse_perm(perm):
                    inverse_users = list(inverse_node.users)
                    if (
                        len(inverse_users) == 1
                        and inverse_users[0].op == "call_function"
                        and str(inverse_users[0].target) == "aten.contiguous.default"
                    ):
                        inverse_users[0].replace_all_uses_with(root_replacement)
                    else:
                        inverse_node.replace_all_uses_with(root_replacement)
                    changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        inverse_perm = _normalize_perm(node.args[1])
        if inverse_perm is None:
            continue
        source = _match_permute_chain_source(node.args[0], _inverse_perm(inverse_perm))
        if source is None:
            continue
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            node_users[0].replace_all_uses_with(source)
        else:
            node.replace_all_uses_with(source)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        cat_node = node.args[0]
        if (
            not isinstance(cat_node, torch.fx.Node)
            or cat_node.op != "call_function"
            or str(cat_node.target) != "aten.cat.default"
            or len(cat_node.args) < 1
        ):
            continue
        cat_inputs_arg = cat_node.args[0]
        if not isinstance(cat_inputs_arg, (list, tuple)):
            continue
        cat_dim = None
        if len(cat_node.args) >= 2:
            cat_dim = int(cat_node.args[1])
        elif "dim" in cat_node.kwargs:
            cat_dim = int(cat_node.kwargs["dim"])
        if cat_dim != 3:
            continue
        folded_inputs = []
        for cat_input in list(cat_inputs_arg):
            folded_source = _match_permute_chain_source(cat_input, [0, 2, 3, 1])
            if folded_source is None:
                folded_inputs = []
                break
            folded_inputs.append(folded_source)
        if len(folded_inputs) != len(list(cat_inputs_arg)):
            continue
        folded_cat_kwargs = dict(cat_node.kwargs)
        folded_cat_args = (folded_inputs, 1)
        if "dim" in folded_cat_kwargs:
            folded_cat_kwargs["dim"] = 1
            folded_cat_args = (folded_inputs,)
        cat_node.args = folded_cat_args
        cat_node.kwargs = folded_cat_kwargs
        cat_node.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            inverse_users[0].replace_all_uses_with(cat_node)
        else:
            node.replace_all_uses_with(cat_node)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        pad_node = node.args[0]
        if not isinstance(pad_node, torch.fx.Node):
            continue
        if (
            pad_node.op != "call_function"
            or str(pad_node.target) != "aten.pad.default"
            or len(pad_node.args) < 4
        ):
            continue
        source = _match_permute_chain_source(pad_node.args[0], [0, 2, 3, 1])
        if source is None:
            continue
        pad_values = list(pad_node.args[1])
        if len(pad_values) != 6:
            continue
        if [int(v) for v in pad_values[:2]] != [0, 0]:
            continue
        cf_pad = [int(pad_values[2]), int(pad_values[3]), int(pad_values[4]), int(pad_values[5])]
        with graph.inserting_before(pad_node):
            folded_pad = graph.call_function(
                pad_node.target,
                args=(source, cf_pad, *tuple(pad_node.args[2:])),
                kwargs=dict(pad_node.kwargs),
            )
        folded_pad.meta = dict(getattr(node, "meta", {}))
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            node_users[0].replace_all_uses_with(folded_pad)
        else:
            node.replace_all_uses_with(folded_pad)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 2, 3, 1]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        node_users = list(node.users)
        if len(node_users) != 1:
            continue
        contiguous_node = node_users[0]
        if (
            contiguous_node.op != "call_function"
            or str(contiguous_node.target) != "aten.contiguous.default"
        ):
            continue
        contiguous_users = list(contiguous_node.users)
        if len(contiguous_users) != 1:
            continue
        sum_node = contiguous_users[0]
        if (
            sum_node.op != "call_function"
            or str(sum_node.target) != "aten.sum.dim_IntList"
            or len(sum_node.args) < 3
            or list(sum_node.args[1]) != [3]
            or bool(sum_node.args[2]) is not True
        ):
            continue
        sum_users = list(sum_node.users)
        if len(sum_users) != 1:
            continue
        sigmoid_node = sum_users[0]
        if (
            sigmoid_node.op != "call_function"
            or str(sigmoid_node.target) != "aten.sigmoid.default"
        ):
            continue
        sigmoid_users = list(sigmoid_node.users)
        if len(sigmoid_users) != 1:
            continue
        inverse_node = sigmoid_users[0]
        if (
            inverse_node.op != "call_function"
            or str(inverse_node.target) != "aten.permute.default"
            or len(inverse_node.args) < 2
            or _normalize_perm(inverse_node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        with graph.inserting_before(node):
            folded_sum = graph.call_function(
                sum_node.target,
                args=(source, [1], True),
                kwargs=dict(sum_node.kwargs),
            )
            folded_sigmoid = graph.call_function(
                sigmoid_node.target,
                args=(folded_sum,),
                kwargs=dict(sigmoid_node.kwargs),
            )
        folded_sum.meta = dict(getattr(sum_node, "meta", {}))
        folded_sigmoid.meta = dict(getattr(inverse_node, "meta", {}))
        inverse_users = list(inverse_node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            inverse_users[0].replace_all_uses_with(folded_sigmoid)
        else:
            inverse_node.replace_all_uses_with(folded_sigmoid)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        source_shape = None
        source_meta_val = getattr(source, "meta", {}).get("val", None)
        if isinstance(source_meta_val, torch.Tensor):
            source_shape = [int(v) for v in list(source_meta_val.shape)]
        elif source.op == "get_attr" and isinstance(source.target, str):
            source_tensor = getattr(graph_module, source.target, None)
            if isinstance(source_tensor, torch.Tensor):
                source_shape = [int(v) for v in list(source_tensor.shape)]
        if source_shape is None or len(source_shape) != 4:
            continue
        non_singleton_axes = [idx for idx, dim in enumerate(source_shape) if int(dim) > 1]
        if len(non_singleton_axes) != 1 or int(non_singleton_axes[0]) != 3:
            continue
        reshaped_shape = [int(source_shape[0]), int(source_shape[3]), int(source_shape[1]), int(source_shape[2])]
        with graph.inserting_before(node):
            folded_reshape = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(source, reshaped_shape),
                kwargs={},
            )
        folded_reshape.meta = dict(getattr(node, "meta", {}))
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            node_users[0].replace_all_uses_with(folded_reshape)
        else:
            node.replace_all_uses_with(folded_reshape)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        mul_node = node.args[0]
        if (
            not isinstance(mul_node, torch.fx.Node)
            or mul_node.op != "call_function"
            or str(mul_node.target) != "aten.mul.Tensor"
            or len(mul_node.args) != 2
        ):
            continue
        mean_node = None
        const_node = None
        for arg in mul_node.args:
            if (
                isinstance(arg, torch.fx.Node)
                and arg.op == "call_function"
                and str(arg.target) == "aten.mean.dim"
                and len(arg.args) >= 3
                and list(arg.args[1]) == [1, 2]
                and bool(arg.args[2]) is True
            ):
                mean_node = arg
            elif isinstance(arg, torch.fx.Node):
                const_node = arg
        if mean_node is None or const_node is None:
            continue
        mean_input = mean_node.args[0]
        if not (
            isinstance(mean_input, torch.fx.Node)
            and mean_input.op == "call_function"
            and str(mean_input.target) == "aten.contiguous.default"
            and len(mean_input.args) >= 1
            and isinstance(mean_input.args[0], torch.fx.Node)
            and mean_input.args[0].op == "call_function"
            and str(mean_input.args[0].target) == "aten.permute.default"
            and len(mean_input.args[0].args) >= 2
            and _normalize_perm(mean_input.args[0].args[1]) == [0, 2, 3, 1]
            and isinstance(mean_input.args[0].args[0], torch.fx.Node)
        ):
            continue
        source = mean_input.args[0].args[0]
        const_shape = None
        const_meta_val = getattr(const_node, "meta", {}).get("val", None)
        if isinstance(const_meta_val, torch.Tensor):
            const_shape = [int(v) for v in list(const_meta_val.shape)]
        if const_node.op == "get_attr" and isinstance(const_node.target, str):
            const_tensor = getattr(graph_module, const_node.target, None)
            if isinstance(const_tensor, torch.Tensor):
                const_shape = [int(v) for v in list(const_tensor.shape)]
        if const_shape is None or len(const_shape) != 4 or const_shape[:3] != [1, 1, 1]:
            continue
        reshaped_const_shape = [1, int(const_shape[3]), 1, 1]
        with graph.inserting_before(mean_node):
            folded_mean = graph.call_function(
                mean_node.target,
                args=(source, [2, 3], True),
                kwargs=dict(mean_node.kwargs),
            )
            folded_const = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(const_node, reshaped_const_shape),
                kwargs={},
            )
            folded_mul = graph.call_function(
                mul_node.target,
                args=(
                    folded_mean if mul_node.args[0] is mean_node else folded_const,
                    folded_const if mul_node.args[0] is mean_node else folded_mean,
                ),
                kwargs=dict(mul_node.kwargs),
            )
        folded_mean.meta = dict(getattr(mean_node, "meta", {}))
        folded_const.meta = dict(getattr(const_node, "meta", {}))
        folded_mul.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            inverse_users[0].replace_all_uses_with(folded_mul)
        else:
            node.replace_all_uses_with(folded_mul)
        changed = True

    binary_targets = {
        "aten.add.Tensor",
        "aten.div.Tensor",
        "aten.mul.Tensor",
        "aten.sub.Tensor",
    }
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        inverse_perm = _normalize_perm(node.args[1])
        if inverse_perm is None:
            continue
        binary_node = node.args[0]
        if not isinstance(binary_node, torch.fx.Node):
            continue
        if (
            binary_node.op != "call_function"
            or str(binary_node.target) not in binary_targets
            or len(binary_node.args) != 2
        ):
            continue
        input_perm = _inverse_perm(inverse_perm)
        lhs_source = _match_binary_input_source(binary_node.args[0], input_perm)
        rhs_source = _match_binary_input_source(binary_node.args[1], input_perm)
        if lhs_source is None or rhs_source is None:
            continue
        with graph.inserting_before(binary_node):
            folded_binary = graph.call_function(
                binary_node.target,
                args=(lhs_source, rhs_source),
                kwargs=dict(binary_node.kwargs),
            )
        folded_binary.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            inverse_users[0].replace_all_uses_with(folded_binary)
        else:
            node.replace_all_uses_with(folded_binary)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

def _fold_layout_preserving_permute_chains(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    changed = False
    unary_targets = {
        "aten.relu.default",
    }
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or list(node.args[1]) != [0, 2, 3, 1]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        permute_users = list(node.users)
        if len(permute_users) != 1:
            continue
        contiguous_node = permute_users[0]
        if (
            contiguous_node.op != "call_function"
            or str(contiguous_node.target) != "aten.contiguous.default"
        ):
            continue
        contiguous_users = list(contiguous_node.users)
        if len(contiguous_users) != 1:
            continue
        unary_node = contiguous_users[0]
        if (
            unary_node.op != "call_function"
            or str(unary_node.target) not in unary_targets
        ):
            continue
        inverse_permute_nodes = list(unary_node.users)
        if len(inverse_permute_nodes) == 0:
            continue
        if any(
            inverse_node.op != "call_function"
            or str(inverse_node.target) != "aten.permute.default"
            or len(inverse_node.args) < 2
            or list(inverse_node.args[1]) != [0, 3, 1, 2]
            for inverse_node in inverse_permute_nodes
        ):
            continue
        with graph.inserting_before(node):
            folded_unary = graph.call_function(
                unary_node.target,
                args=(source, *tuple(unary_node.args[1:])),
                kwargs=dict(unary_node.kwargs),
            )
        folded_unary.meta = dict(getattr(source, "meta", {}))
        for inverse_node in inverse_permute_nodes:
            inverse_users = list(inverse_node.users)
            if (
                len(inverse_users) == 1
                and inverse_users[0].op == "call_function"
                and str(inverse_users[0].target) == "aten.contiguous.default"
            ):
                inverse_users[0].replace_all_uses_with(folded_unary)
            else:
                inverse_node.replace_all_uses_with(folded_unary)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

with torch.no_grad():
    exported = torch.export.export(model, example_inputs)
exported = _prune_alias_nodes(exported)
exported = _fold_inverse_permute_round_trips(exported)
exported = _fold_layout_preserving_permute_chains(exported)
torch.export.save(exported, str(exported_program_path))
print(json.dumps({"file_name": exported_program_path.name}))
"""
    child_payload, last_error_message = _run_generated_package_export_child(
        example_inputs=example_inputs,
        child_script=child_script,
        package_path=package_path,
        artifact_path=exported_program_path,
        child_payload={},
        temp_prefix="onnx2tf_exported_program_",
    )
    if child_payload is None or not exported_program_path.exists():
        _remove_generated_package_artifact_if_exists(exported_program_path)
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message or "torch.export.save did not produce an artifact.",
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "ExportedProgram export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        return None
    try:
        _strip_stack_traces_from_exported_program_archive(exported_program_path)
    except Exception as ex:
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "ExportedProgram archive cleanup failed for the generated native PyTorch package. "
                f"package_dir={package_dir} artifact={exported_program_path} details={ex}"
            ) from ex
        last_error_message = str(ex)
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="exported_program",
        file_name=str(child_payload.get("file_name", exported_program_file_name)),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
    )
    return str(exported_program_path)


def _strip_stack_traces_from_exported_program_archive(exported_program_path: Path) -> None:
    archive_path = Path(exported_program_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"ExportedProgram archive not found. path={archive_path}")
    with tempfile.NamedTemporaryFile(
        prefix="onnx2tf_exported_program_strip_",
        suffix=".pt2",
        delete=False,
        dir=str(archive_path.parent),
    ) as tmp_file:
        temp_archive_path = Path(tmp_file.name)
    try:
        removed_count = 0
        with zipfile.ZipFile(str(archive_path), "r") as source_archive, zipfile.ZipFile(
            str(temp_archive_path),
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as stripped_archive:
            for info in source_archive.infolist():
                payload = source_archive.read(info.filename)
                if info.filename.endswith("models/model.json"):
                    model_json = json.loads(payload)

                    def _strip_stack_trace_fields(value: Any) -> None:
                        nonlocal removed_count
                        if isinstance(value, dict):
                            if "stack_trace" in value:
                                del value["stack_trace"]
                                removed_count += 1
                            for child in value.values():
                                _strip_stack_trace_fields(child)
                            return
                        if isinstance(value, list):
                            for child in value:
                                _strip_stack_trace_fields(child)

                    _strip_stack_trace_fields(model_json)
                    payload = json.dumps(model_json, separators=(",", ":")).encode("utf-8")
                stripped_archive.writestr(info, payload)
        if removed_count == 0:
            temp_archive_path.unlink(missing_ok=True)
            return
        temp_archive_path.replace(archive_path)
    except Exception:
        temp_archive_path.unlink(missing_ok=True)
        raise


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


def _make_unique_identifier(base_name: str, used_names: Set[str]) -> str:
    candidate = str(base_name)
    suffix = 1
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


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


def _preferred_reshape_target_values(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None:
        return None
    if tensor.shape_signature is not None:
        signature = [int(v) for v in list(tensor.shape_signature)]
        if len(signature) == len(list(tensor.shape)) and any(int(v) <= 0 for v in signature):
            return signature
    return [int(v) for v in list(tensor.shape)]


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


def _read_onnx_transpose_perm(node: Any) -> Optional[List[int]]:
    if str(getattr(node, "op_type", "")) != "Transpose":
        return None
    for attr in list(getattr(node, "attribute", [])):
        if str(getattr(attr, "name", "")) != "perm":
            continue
        try:
            values = onnx.helper.get_attribute_value(attr)
        except Exception:
            return None
        try:
            return [int(v) for v in list(values)]
        except Exception:
            return None
        return None


def _is_onnx_boundary_layout_passthrough_node(
    *,
    node: Any,
    source_tensor_name: str,
) -> bool:
    passthrough_op_types = {
        "Abs",
        "Add",
        "Cast",
        "Clip",
        "Div",
        "Identity",
        "LeakyRelu",
        "Mul",
        "Relu",
        "Sigmoid",
        "Softmax",
        "Sub",
        "Tanh",
    }
    if str(getattr(node, "op_type", "")) not in passthrough_op_types:
        return False
    inputs = [str(v) for v in list(getattr(node, "input", []))]
    outputs = [str(v) for v in list(getattr(node, "output", []))]
    return len(outputs) == 1 and str(source_tensor_name) in set(inputs)


def _infer_public_layouts_from_onnx_graph(reference_onnx_graph: Any) -> Dict[str, str]:
    graph = getattr(reference_onnx_graph, "graph", None)
    if graph is None:
        return {}
    consumers: Dict[str, List[Any]] = {}
    producer_by_output: Dict[str, Any] = {}
    for node in list(graph.node):
        for output_name in list(getattr(node, "output", [])):
            producer_by_output[str(output_name)] = node
        for input_name in list(getattr(node, "input", [])):
            consumers.setdefault(str(input_name), []).append(node)

    def _walk_input_boundary(tensor_name: str, rank: int) -> Optional[str]:
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            user_nodes = consumers.get(current_tensor_name, [])
            if len(user_nodes) != 1:
                return None
            node = user_nodes[0]
            perm = _read_onnx_transpose_perm(node)
            if perm == _perm_cl_to_cf(rank):
                return channel_last_logical_layout(rank)
            if perm == _perm_cf_to_cl(rank):
                return channel_first_logical_layout(rank)
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=current_tensor_name,
            ):
                return None
            current_tensor_name = str(list(getattr(node, "output", []))[0])
        return None

    def _walk_output_boundary(tensor_name: str, rank: int) -> Optional[str]:
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            node = producer_by_output.get(current_tensor_name, None)
            if node is None:
                return None
            perm = _read_onnx_transpose_perm(node)
            if perm == _perm_cf_to_cl(rank):
                return channel_last_logical_layout(rank)
            if perm == _perm_cl_to_cf(rank):
                return channel_first_logical_layout(rank)
            inputs = [str(v) for v in list(getattr(node, "input", []))]
            if len(inputs) != 1:
                return None
            previous_tensor_name = inputs[0]
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=previous_tensor_name,
            ):
                return None
            current_tensor_name = previous_tensor_name
        return None

    public_layout_map: Dict[str, str] = {}
    for value_info in list(graph.input):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        rank = len(dims)
        if rank not in {3, 4, 5}:
            continue
        inferred_layout = _walk_input_boundary(tensor_name, rank)
        if inferred_layout is not None:
            public_layout_map[tensor_name] = inferred_layout
    for value_info in list(graph.output):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        rank = len(dims)
        if rank not in {3, 4, 5}:
            continue
        inferred_layout = _walk_output_boundary(tensor_name, rank)
        if inferred_layout is not None:
            public_layout_map[tensor_name] = inferred_layout
    return public_layout_map


def _infer_batchless_rank3_image_boundaries_from_onnx_graph(
    reference_onnx_graph: Any,
) -> Set[str]:
    graph = getattr(reference_onnx_graph, "graph", None)
    if graph is None:
        return set()
    consumers: Dict[str, List[Any]] = {}
    producer_by_output: Dict[str, Any] = {}
    for node in list(graph.node):
        for output_name in list(getattr(node, "output", [])):
            producer_by_output[str(output_name)] = node
        for input_name in list(getattr(node, "input", [])):
            consumers.setdefault(str(input_name), []).append(node)

    def _input_is_batchless_channel_first_image(tensor_name: str, rank: int) -> bool:
        if rank != 3:
            return False
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            user_nodes = consumers.get(current_tensor_name, [])
            if len(user_nodes) != 1:
                return False
            node = user_nodes[0]
            axes = _read_onnx_unsqueeze_axes(node)
            if axes == [0]:
                return True
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=current_tensor_name,
            ):
                return False
            current_tensor_name = str(list(getattr(node, "output", []))[0])
        return False

    def _output_is_batchless_channel_first_image(tensor_name: str, rank: int) -> bool:
        if rank != 3:
            return False
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            node = producer_by_output.get(current_tensor_name, None)
            if node is None:
                return False
            axes = _read_onnx_squeeze_axes(node)
            if axes == [0]:
                return True
            inputs = [str(v) for v in list(getattr(node, "input", []))]
            if len(inputs) != 1:
                return False
            previous_tensor_name = inputs[0]
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=previous_tensor_name,
            ):
                return False
            current_tensor_name = previous_tensor_name
        return False

    boundary_names: Set[str] = set()
    for value_info in list(graph.input):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        if _input_is_batchless_channel_first_image(tensor_name, len(dims)):
            boundary_names.add(tensor_name)
    for value_info in list(graph.output):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        if _output_is_batchless_channel_first_image(tensor_name, len(dims)):
            boundary_names.add(tensor_name)
    return boundary_names


def _merge_reference_public_boundary_metadata(
    *,
    imported_model_ir: ModelIR,
    reference_model_ir: Optional[ModelIR],
    reference_onnx_graph: Optional[Any] = None,
) -> None:
    if reference_model_ir is None:
        return
    boundary_shape_map = reference_model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
    public_layout_map = reference_model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    onnx_graph_public_layout_map = (
        _infer_public_layouts_from_onnx_graph(reference_onnx_graph)
        if reference_onnx_graph is not None
        else {}
    )
    batchless_rank3_boundary_names = (
        _infer_batchless_rank3_image_boundaries_from_onnx_graph(reference_onnx_graph)
        if reference_onnx_graph is not None
        else set()
    )
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
    imported_model_ir.inputs = [str(v) for v in list(reference_model_ir.inputs)]
    imported_model_ir.outputs = [str(v) for v in list(reference_model_ir.outputs)]

    desired_public_layout_map: Dict[str, str] = {}
    desired_public_shape_map: Dict[str, List[int]] = {}
    for tensor_name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs):
        ref_tensor = reference_model_ir.tensors.get(str(tensor_name), None)
        if ref_tensor is None:
            continue
        desired_public_shape_map[str(tensor_name)] = [
            int(v) for v in list(ref_tensor.shape_signature or ref_tensor.shape)
        ]
        desired_layout = normalize_logical_layout(
            onnx_graph_public_layout_map.get(
                str(tensor_name),
                public_layout_map.get(str(tensor_name), ref_tensor.logical_layout),
            )
        )
        if recurrent_public_boundary_context and len(list(ref_tensor.shape)) == 3:
            desired_layout = "NWC"
        desired_public_layout_map[str(tensor_name)] = desired_layout

    _ensure_public_boundary_layout_bridges(
        model_ir=imported_model_ir,
        desired_public_shape_map=desired_public_shape_map,
        desired_public_layout_map=desired_public_layout_map,
    )

    for tensor_name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs):
        ref_tensor = reference_model_ir.tensors.get(str(tensor_name), None)
        imported_tensor = imported_model_ir.tensors.get(str(tensor_name), None)
        if ref_tensor is None or imported_tensor is None:
            continue
        imported_tensor.shape_signature = [int(v) for v in list(ref_tensor.shape_signature or ref_tensor.shape)]
        imported_tensor.logical_layout = desired_public_layout_map.get(
            str(tensor_name),
            normalize_logical_layout(ref_tensor.logical_layout),
        )
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
                onnx_graph_public_layout_map.get(
                    str(name),
                    public_layout_map.get(
                        str(name),
                        reference_model_ir.tensors[str(name)].logical_layout,
                    ),
                )
            )
        )
        for name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
        if str(name) in reference_model_ir.tensors
    }
    imported_model_ir.metadata["batchless_rank3_public_boundary_names"] = sorted(
        str(name)
        for name in list(batchless_rank3_boundary_names)
        if str(name) in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
    )


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
    fallback_saved_model_factory: Optional[Callable[[], Optional[str]]] = None,
    fallback_tflite_has_custom_ops: bool = False,
) -> str:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "PyTorch export requires `torch` to be installed."
        ) from ex

    resolved_fallback_saved_model_path = (
        str(fallback_saved_model_path)
        if fallback_saved_model_path is not None and str(fallback_saved_model_path).strip() != ""
        else None
    )

    def _get_fallback_saved_model_path() -> Optional[str]:
        nonlocal resolved_fallback_saved_model_path
        if resolved_fallback_saved_model_path is not None:
            return resolved_fallback_saved_model_path
        if fallback_saved_model_factory is None:
            return None
        try:
            generated_path = fallback_saved_model_factory()
        except Exception:
            return None
        if generated_path is None or str(generated_path).strip() == "":
            return None
        resolved_fallback_saved_model_path = str(generated_path)
        return resolved_fallback_saved_model_path

    model_op_types = {str(op.op_type) for op in model_ir.operators}
    control_or_recurrent_ops = {
        "WHILE",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if (
        fallback_tflite_path is not None
        and str(fallback_tflite_path).strip() != ""
        and not bool(fallback_tflite_has_custom_ops)
        and any(op_type in control_or_recurrent_ops for op_type in model_op_types)
    ):
        try:
            imported_native_package_path = _try_export_native_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
                reference_model_ir=model_ir,
                reference_onnx_graph=fallback_onnx_graph,
            )
            if imported_native_package_path is not None:
                return imported_native_package_path
        except Exception:
            pass

    try:
        normalized: Optional[ModelIR] = None
        native_prep_error: Optional[Exception] = None

        try:
            normalized = prepare_model_ir_for_native_pytorch(model_ir)
            _ensure_no_custom_ops(normalized)
            _ensure_supported_ops(normalized)
        except Exception as ex:
            normalized = None
            native_prep_error = ex
        if (
            normalized is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and not bool(fallback_tflite_has_custom_ops)
        ):
            try:
                imported_native_package_path = _try_export_native_package_from_tflite_import(
                    output_folder_path=output_folder_path,
                    fallback_tflite_path=str(fallback_tflite_path),
                    reference_model_ir=model_ir,
                    reference_onnx_graph=fallback_onnx_graph,
                )
                if imported_native_package_path is not None:
                    return imported_native_package_path
            except Exception:
                pass
        fallback_saved_model_path_for_export = _get_fallback_saved_model_path() if normalized is None else resolved_fallback_saved_model_path
        if (
            normalized is None
            and fallback_saved_model_path_for_export is not None
            and _should_prefer_saved_model_backed_package(model_ir)
        ):
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path_for_export),
            )
        if (
            normalized is None
            and fallback_saved_model_path is None
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
            normalized is None
            and fallback_tflite_path is not None
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

        if normalized is None:
            if native_prep_error is not None:
                raise native_prep_error
            raise ModelIRPyTorchExportError(
                "Native PyTorch export preparation failed for an unknown reason."
            )
        tensor_storage_name_map = _make_tensor_storage_name_map(normalized)

        os.makedirs(output_folder_path, exist_ok=True)
        metadata = _build_metadata_payload(normalized)
        metadata["execution_backend"] = "native"
        metadata["tensor_storage_names"] = dict(tensor_storage_name_map)
        native_load_specs: Optional[List[Tuple[str, str]]] = None
        try:
            native_load_specs = _write_native_model_file(
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
            _write_generated_package_common_files(output_folder_path)
            _write_wrapper_model_file(output_folder_path)
            metadata["execution_backend"] = "runtime_wrapper"
        metadata_path = os.path.join(output_folder_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if native_load_specs is not None:
            state_dict = _build_native_generated_state_dict(
                package_path=output_folder_path,
                model_ir=normalized,
                load_specs=native_load_specs,
            )
        else:
            state_dict = {}
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
        fallback_saved_model_path_for_export = _get_fallback_saved_model_path()
        if fallback_saved_model_path_for_export is not None:
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path_for_export),
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
                imported_native_package_path = _try_export_native_package_from_tflite_import(
                    output_folder_path=output_folder_path,
                    fallback_tflite_path=str(fallback_tflite_path),
                    reference_model_ir=model_ir,
                    reference_onnx_graph=fallback_onnx_graph,
                )
                if imported_native_package_path is not None:
                    return imported_native_package_path
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
