from __future__ import annotations

from typing import Callable, Collection, Dict, List, Optional, Set

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    is_channel_first_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _shape_lists_equal_relaxed,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cf_to_cl,
    _read_transpose_perm,
)


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
                channel_first_tensor_expr_aliases=(
                    channel_first_tensor_expr_aliases
                ),
                direct_codegen_unary_expressions=(
                    direct_codegen_unary_expressions
                ),
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
    if (
        producer_type in direct_codegen_unary_expressions
        and len(producer_op.inputs) == 1
    ):
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
