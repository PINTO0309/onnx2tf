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


from .common import ModelIRPyTorchExportError, _clone_tensor, _collect_model_op_types, _make_unique_identifier
from .layout_normalization import _align_public_boundary_shapes_to_onnx_contract, _ensure_public_boundary_layout_bridges, _is_layout_agnostic_native_model_ir, normalize_model_ir_for_pytorch_channel_first
from .native_codegen import _DIRECT_CODEGEN_SUPPORTED_OP_TYPES, _can_direct_codegen_sequence_lstm_op, _can_direct_codegen_sequence_rnn_op

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
