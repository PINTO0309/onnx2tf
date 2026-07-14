from __future__ import annotations

from typing import Callable, Collection, Dict, List, Optional, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _broadcast_shapes_relaxed,
    _is_all_ones_shape,
    _shape_can_broadcast_to_target_relaxed,
    _shape_lists_equal_relaxed,
)
from onnx2tf.tflite_builder.pytorch_graph_policy import (
    _channel_first_shape_values_for_model_ir,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cl_to_cf,
    _read_transpose_perm,
)


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
    if str(lhs_name) in runtime_shape_uncertain_tensors and _is_all_ones_shape(
        rhs_shape
    ):
        return "lhs"
    if str(rhs_name) in runtime_shape_uncertain_tensors and _is_all_ones_shape(
        lhs_shape
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
    lhs_layout = normalize_logical_layout(lhs_tensor.logical_layout)
    rhs_layout = normalize_logical_layout(rhs_tensor.logical_layout)
    lhs_cf_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(lhs_name),
    )
    rhs_cf_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(rhs_name),
    )
    output_cf_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(output_name),
    )
    if (
        lhs_cf_shape is not None
        and rhs_cf_shape is not None
        and output_cf_shape is not None
        and len(lhs_cf_shape) == len(rhs_cf_shape) == len(output_cf_shape)
        and len(lhs_cf_shape) in {3, 4, 5}
        and is_channel_first_logical_layout(lhs_layout)
        and is_channel_first_logical_layout(rhs_layout)
    ):
        try:
            broadcast_cf_shape = [
                int(v)
                for v in list(
                    np.broadcast_shapes(tuple(lhs_cf_shape), tuple(rhs_cf_shape))
                )
            ]
            if [int(v) for v in list(broadcast_cf_shape)] == [
                int(v) for v in list(output_cf_shape)
            ]:
                return False
        except Exception:
            pass
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
        broadcast_shape = [
            int(v)
            for v in list(
                np.broadcast_shapes(tuple(lhs_shape), tuple(rhs_shape))
            )
        ]
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
    if len(lhs_signature) != len(rhs_signature) or len(lhs_signature) != len(
        output_signature
    ):
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
    channel_first_reduction_plan_fn: Callable[
        [OperatorIR, str], Optional[Tuple[str, List[int]]]
    ],
    can_emit_channel_first_shape_preserving_unary_op_fn: Callable[
        [OperatorIR], bool
    ],
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
            # Rank-3 feature-last tensors still need a materialized alias here.
            # Later transpose emission cannot always reuse the channel-first
            # producer alias because `_tensor_expr` intentionally preserves the
            # logical NWC name for public/bridge cases.
            if int(output_rank) == 3:
                return False
            continue
        if consumer_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            if len(consumer_op.inputs) < 2 or str(consumer_op.inputs[0]) != str(
                output_name
            ):
                return False
            input_pre_permute = conv2d_input_pre_permute_fn(
                tensor_shape_list_fn(str(consumer_op.inputs[0])),
                tensor_shape_list_fn(str(consumer_op.outputs[0])),
                tensor_shape_list_fn(str(consumer_op.inputs[1])),
                consumer_op.options,
                input_logical_layout=normalize_logical_layout(
                    output_tensor.logical_layout
                ),
                output_logical_layout=normalize_logical_layout(
                    model_ir.tensors[str(consumer_op.outputs[0])].logical_layout
                ),
                depthwise=(consumer_type == "DEPTHWISE_CONV_2D"),
            )
            if list(input_pre_permute or []) != list(expected_input_bridge_perm):
                return False
            continue
        if consumer_type in {
            "SUM",
            "MEAN",
            "REDUCE_MAX",
            "REDUCE_MIN",
            "REDUCE_PROD",
            "REDUCE_ANY",
        }:
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
            if (
                consumer_output_tensor is None
                or len(list(consumer_output_tensor.shape)) != output_rank
            ):
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
            if not _shape_can_broadcast_to_target_relaxed(
                current_shape, consumer_output_shape
            ):
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
                if not _shape_can_broadcast_to_target_relaxed(
                    input_shape, consumer_output_shape
                ):
                    return False
                broadcast_shape = (
                    list(input_shape)
                    if broadcast_shape is None
                    else _broadcast_shapes_relaxed(broadcast_shape, input_shape)
                )
                if broadcast_shape is None:
                    return False
            if not _shape_lists_equal_relaxed(
                broadcast_shape, consumer_output_shape
            ):
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
