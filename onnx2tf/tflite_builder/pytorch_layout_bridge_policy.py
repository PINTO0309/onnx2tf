from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import _constant_int_list
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _compose_axis_permutations,
    _is_standard_channel_layout_permutation,
    _read_transpose_perm,
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
    if (
        str(producer_op.op_type) != "TRANSPOSE"
        or len(producer_op.outputs) != 1
        or len(producer_op.inputs) < 1
    ):
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
    if required_output_layout is not None and output_layout != normalize_logical_layout(
        required_output_layout
    ):
        return None
    expected_perm = logical_layout_permutation(
        source_layout=input_layout,
        target_layout=output_layout,
    )
    actual_perm = _read_transpose_perm(model_ir, bridge_op)
    if expected_perm is None or actual_perm is None:
        return None
    if [int(v) for v in list(expected_perm)] != [
        int(v) for v in list(actual_perm)
    ]:
        return None
    return output_name, bridge_op_idx


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
